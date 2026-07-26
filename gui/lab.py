"""Shared helpers for the GUI — the only thing that reads the lab from disk.

Every view imports from here. Nothing in the GUI reimplements lab logic; it
discovers experiments, infobase notes, and runs scripts/tests in the venv.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = LAB_ROOT / "experiments"
INFOBASE = LAB_ROOT / "infobase"
HEAVY_EXPERIMENT_IDS = {"e21"}
SUPPORTED_LANGS = {"tr"}

# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    tier: str          # e.g. "tier0_neuron"
    folder: str        # e.g. "e01_single_neuron"
    path: Path

    @property
    def id(self) -> str:
        return self.folder.split("_", 1)[0]            # "e01"

    @property
    def name(self) -> str:
        rest = self.folder.split("_", 1)
        return rest[1].replace("_", " ") if len(rest) > 1 else self.folder

    @property
    def label(self) -> str:
        return f"{self.id} · {self.name}"

    @property
    def is_heavy(self) -> bool:
        return self.id in HEAVY_EXPERIMENT_IDS

    @property
    def readme(self) -> Path:
        return self.path / "README.md"

    @property
    def run_py(self) -> Path:
        return self.path / "run.py"

    @property
    def notes(self) -> Path:
        return self.path / "notes.md"


def tier_label(tier_dir: str) -> str:
    m = re.match(r"tier(\d+)_(.+)", tier_dir)
    return f"Tier {m.group(1)} · {m.group(2)}" if m else tier_dir


def list_tiers() -> dict[str, list[Experiment]]:
    """Ordered {tier_dir: [Experiment, ...]} for tiers that contain experiments."""
    tiers: dict[str, list[Experiment]] = {}
    for tdir in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
        exps = [
            Experiment(tdir.name, e.name, e)
            for e in sorted(tdir.iterdir())
            if e.is_dir() and (e / "run.py").exists()
        ]
        if exps:
            tiers[tdir.name] = exps
    return tiers


def all_experiments() -> list[Experiment]:
    return [e for exps in list_tiers().values() for e in exps]


# ---------------------------------------------------------------------------
# Running scripts / tests in the venv
# ---------------------------------------------------------------------------


def run_script(path: Path, args: list[str] | None = None, timeout: int = 60):
    """Run a python script with the current interpreter. Returns (code, output)."""
    cmd = [sys.executable, str(path), *(args or [])]
    env = {"MPLBACKEND": "Agg"}  # never pop a window from a subprocess
    try:
        proc = subprocess.run(
            cmd,
            cwd=LAB_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**_base_env(), **env},
        )
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"(timed out after {timeout}s)"


def run_pytest(timeout: int = 120):
    cmd = [sys.executable, "-m", "pytest", "-q"]
    try:
        proc = subprocess.run(
            cmd, cwd=LAB_ROOT, capture_output=True, text=True, timeout=timeout,
            env=_base_env(),
        )
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"(pytest timed out after {timeout}s)"


def _base_env():
    import os

    # Put the lab root on PYTHONPATH so subprocess `python run.py` can `import core`
    # even where the package isn't pip-installed editable (e.g. the server deploy).
    env = dict(os.environ)
    env["PYTHONPATH"] = str(LAB_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def status_of(exp: Experiment, timeout: int = 30, include_heavy: bool = True) -> str:
    """Return done / todo / error, or heavy when a costly experiment is skipped."""
    if exp.is_heavy and not include_heavy:
        return "heavy"

    code, out = run_script(exp.run_py, timeout=timeout)
    if code == 0:
        return "done"
    if "NotImplementedError" in out:
        return "todo"
    return "error"


# ---------------------------------------------------------------------------
# Infobase
# ---------------------------------------------------------------------------


def list_infobase() -> list[Path]:
    return sorted(p for p in INFOBASE.rglob("*.md") if not is_localized_variant(p))


def is_localized_variant(path: Path) -> bool:
    return any(path.name.endswith(f".{language}{path.suffix}") for language in SUPPORTED_LANGS)


def read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def localized_path(path: Path, language: str) -> Path:
    if language == "en":
        return path
    localized = path.with_name(f"{path.stem}.{language}{path.suffix}")
    return localized if localized.exists() else path


def read_localized(path: Path, language: str) -> str:
    return read(localized_path(path, language))
