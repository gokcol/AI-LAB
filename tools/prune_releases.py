"""Remove obsolete AI Lab builds after a healthy activation, under the deploy lock."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import sys


RELEASE_ID = re.compile(r"\d{8}T\d{6}Z")
ARCHIVE = re.compile(r"ai-lab-(\d{8}T\d{6}Z)\.(?:tar\.gz|sha256)")


def retained_releases(base: Path, active: str, rollback: str) -> tuple[Path, Path]:
    """Fail closed if the deployment changed or rollback metadata is unsafe."""
    if not all(RELEASE_ID.fullmatch(name) for name in (active, rollback)) or active == rollback:
        raise ValueError("Expected distinct timestamp IDs for active and rollback releases")
    releases = base / "releases"
    if releases.is_symlink() or releases.resolve(strict=True) != releases:
        raise ValueError("Release storage must be a real directory")
    keep = (releases / active, releases / rollback)
    for path in keep:
        if path.is_symlink() or not path.is_dir() or not (path / ".validated").is_file():
            raise ValueError(f"Retained release is missing or not validated: {path}")
        if not (path / ".venv/bin/python").exists():
            raise ValueError(f"Retained release has no Python runtime: {path}")
    current = base / "current"
    if not current.is_symlink() or current.resolve(strict=True) != keep[0]:
        raise ValueError("Active release changed; cleanup cancelled")
    previous = keep[0] / "previous-release.txt"
    if previous.is_symlink() or previous.read_text().strip() != str(keep[1]):
        raise ValueError("Rollback metadata changed; cleanup cancelled")
    return keep


def check_dependencies(candidates: list[Path], keep: tuple[Path, Path], proc: Path) -> None:
    """Never remove mounted, running, or linked dependencies of retained releases."""
    if not candidates:
        return

    def within_candidate(path: Path) -> bool:
        return any(path == old or old in path.parents for old in candidates)

    for retained in keep:
        for folder, dirs, files in os.walk(retained, followlinks=False):
            for name in dirs + files:
                path = Path(folder) / name
                if path.is_symlink() and within_candidate(path.resolve()):
                    raise ValueError(f"Retained release depends on an older build: {path}")
    for line in (proc / "self/mountinfo").read_text().splitlines():
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4])
        if within_candidate(Path(mount)):
            raise ValueError(f"Release contains a mount: {mount}")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().split(b"\0")
            cwd = Path(os.readlink(entry / "cwd"))
        except (FileNotFoundError, ProcessLookupError):
            continue  # Processes can exit while being inspected; kernel threads lack cwd.
        args = [arg.decode(errors="replace") for arg in cmd]
        if within_candidate(cwd) or any(
            arg == str(old) or arg.startswith(str(old) + "/")
            for arg in args for old in candidates
        ):
            raise ValueError(f"Older build is still used by process {entry.name}")


def prune(base: Path, active: str, rollback: str, *, archives: Path = Path("/tmp"),
          proc: Path = Path("/proc"), apply: bool = False) -> dict[str, list[str]]:
    """Caller holds .release.lock. Preview by default; never follow directory links."""
    base = base.resolve(strict=True)
    keep = retained_releases(base, active, rollback)
    candidates = []
    for path in sorted((base / "releases").iterdir()):
        # Leave newer, not-yet-activated staging builds for their deployment.
        if not RELEASE_ID.fullmatch(path.name) or path in keep or path.name >= active:
            continue
        if path.is_symlink() or not path.is_dir() or not (path / "gui/app.py").is_file():
            raise ValueError(f"Unrecognized release directory: {path}")
        if (path / "feedback").exists() and not (path / "feedback").is_symlink():
            raise ValueError(f"Release contains unshared feedback: {path}")
        candidates.append(path)
    check_dependencies(candidates, keep, proc)
    old_archives = []
    for path in sorted(archives.iterdir()):
        match = ARCHIVE.fullmatch(path.name)
        # Retain packages for both kept releases, including restaged packages
        # whose archive timestamp can be later than their release directory.
        if match and match[1] < min(active, rollback) and path.is_file() and not path.is_symlink():
            old_archives.append(path)
    if apply:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError("This platform lacks safe directory removal")
        for path in candidates + old_archives:
            retained_releases(base, active, rollback)
            if path in candidates:
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"Release changed during cleanup: {path}")
                shutil.rmtree(path)
            else:
                path.unlink()
    return {"releases": [p.name for p in candidates], "archives": [p.name for p in old_archives]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/opt/ai-lab"))
    parser.add_argument("--active", required=True)
    parser.add_argument("--rollback", required=True)
    parser.add_argument("--apply", action="store_true", help="Delete the planned obsolete builds")
    parser.add_argument("--lock-fd", type=int, help="Inherited deployment-lock descriptor")
    args = parser.parse_args()
    try:
        lock_path = args.base / ".release.lock"
        lock = open(lock_path, "a") if args.lock_fd is None else os.fdopen(os.dup(args.lock_fd), "a")
        with lock:
            if not os.path.samestat(os.fstat(lock.fileno()), lock_path.stat()):
                raise ValueError("Wrong deployment lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = prune(args.base, args.active, args.rollback, apply=args.apply)
        verb = "Removed" if args.apply else "Would remove"
        print(f"{verb} {len(result['releases'])} old releases and {len(result['archives'])} build archives; "
              f"kept active {args.active} and rollback {args.rollback}")
        for name in result["releases"]:
            print(f"  {name}")
    except (OSError, ValueError) as exc:
        sys.exit(f"Release cleanup stopped: {exc}")


if __name__ == "__main__":
    main()
