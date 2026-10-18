"""Old builds are pruned only after activation, without losing a usable rollback."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from tools import prune_releases as cleanup


ROOT = Path(__file__).resolve().parents[1]
OLD = "20261001T010000Z"
ROLLBACK = "20261002T010000Z"
UNUSED = "20261003T010000Z"
ACTIVE = "20261004T010000Z"
STAGED = "20261005T010000Z"


def make_release(base: Path, name: str) -> Path:
    release = base / "releases" / name
    (release / "gui").mkdir(parents=True)
    (release / "gui/app.py").write_text("# A release\n")
    (release / ".venv/bin").mkdir(parents=True)
    (release / ".venv/bin/python").touch()
    (release / ".validated").touch()
    return release


@pytest.fixture
def deployment(tmp_path):
    # macOS's /var aliases /private/var; production paths are already canonical.
    root = tmp_path.resolve()
    base, archives, proc = root / "ai-lab", root / "archives", root / "proc"
    archives.mkdir()
    (proc / "self").mkdir(parents=True)
    (proc / "self/mountinfo").write_text("")
    for name in (OLD, ROLLBACK, UNUSED, ACTIVE, STAGED):
        make_release(base, name)
    (base / "current").symlink_to(base / "releases" / ACTIVE)
    (base / "releases" / ACTIVE / "previous-release.txt").write_text(
        str(base / "releases" / ROLLBACK) + "\n"
    )
    shared = base / "shared/feedback"
    shared.mkdir(parents=True)
    (shared / "learner.json").write_text('{"message":"keep me"}')
    (base / "releases" / OLD / "feedback").symlink_to(shared)
    return base, archives, proc


def prune(deployment, **kwargs):
    base, archives, proc = deployment
    return cleanup.prune(base, ACTIVE, ROLLBACK, archives=archives, proc=proc, **kwargs)


def assert_old_builds_remain(deployment):
    base, _, _ = deployment
    assert (base / "releases" / OLD / "gui/app.py").is_file()
    assert (base / "releases" / UNUSED / "gui/app.py").is_file()


def test_preview_then_apply_keeps_recorded_rollback_and_shared_data(deployment):
    base, archives, _ = deployment
    # The rollback is deliberately not the newest build preceding ACTIVE.
    (base / "releases/manual-backup").mkdir()
    (base / "releases/readme.txt").write_text("Not a release")
    remove = [f"ai-lab-{OLD}.tar.gz", f"ai-lab-{OLD}.sha256"]
    keep = [
        f"ai-lab-{ROLLBACK}.tar.gz", f"ai-lab-{ACTIVE}.sha256",
        f"ai-lab-{UNUSED}.tar.gz", f"ai-lab-{STAGED}.tar.gz",
        "unrelated.tar.gz", f"ai-lab-{OLD}.tar.gz.backup", "ai-lab-invalid.sha256",
    ]
    for name in remove + keep:
        (archives / name).write_text(name)
    external = archives.parent / "external.tar.gz"
    external.write_text("Do not follow archive symlinks")
    symlink_name = "ai-lab-20260901T010000Z.tar.gz"
    (archives / symlink_name).symlink_to(external)

    plan = prune(deployment)
    assert plan == {"releases": [OLD, UNUSED], "archives": sorted(remove)}
    assert_old_builds_remain(deployment)
    assert all((archives / name).is_file() for name in remove)

    assert prune(deployment, apply=True) == plan
    assert sorted(p.name for p in (base / "releases").iterdir()) == sorted(
        [ROLLBACK, ACTIVE, STAGED, "manual-backup", "readme.txt"]
    )
    assert all(not (archives / name).exists() for name in remove)
    assert all((archives / name).is_file() for name in keep)
    assert (archives / symlink_name).is_symlink()
    assert external.read_text() == "Do not follow archive symlinks"
    assert (base / "shared/feedback/learner.json").read_text() == '{"message":"keep me"}'
    assert (base / "current").resolve() == base / "releases" / ACTIVE
    assert prune(deployment, apply=True) == {"releases": [], "archives": []}


@pytest.mark.parametrize("damage", [
    "missing_rollback", "unvalidated_rollback", "missing_runtime", "missing_metadata",
    "malformed_metadata", "metadata_symlink", "changed_current", "current_not_link",
])
def test_invalid_retained_release_metadata_stops_before_deletion(deployment, damage):
    base, _, _ = deployment
    previous = base / "releases" / ACTIVE / "previous-release.txt"
    rollback = base / "releases" / ROLLBACK
    if damage == "missing_rollback":
        rollback.rename(base / "missing-rollback")
    elif damage == "unvalidated_rollback":
        (rollback / ".validated").unlink()
    elif damage == "missing_runtime":
        (rollback / ".venv/bin/python").unlink()
    elif damage == "missing_metadata":
        previous.unlink()
    elif damage == "malformed_metadata":
        previous.write_text("../../outside\n")
    elif damage == "metadata_symlink":
        destination = base / "metadata.txt"
        previous.rename(destination)
        previous.symlink_to(destination)
    else:
        (base / "current").unlink()
        if damage == "changed_current":
            (base / "current").symlink_to(rollback)
        else:
            (base / "current").mkdir()
    with pytest.raises((OSError, ValueError)):
        prune(deployment, apply=True)
    assert_old_builds_remain(deployment)


@pytest.mark.parametrize("active,rollback", [
    ("../outside", ROLLBACK), (ACTIVE, "../outside"), (ACTIVE, ACTIVE),
    (ACTIVE, ""), (ACTIVE, "20261002T010000Z-extra"),
])
def test_rejects_unsafe_or_duplicate_release_ids(deployment, active, rollback):
    base, archives, proc = deployment
    with pytest.raises(ValueError):
        cleanup.prune(base, active, rollback, archives=archives, proc=proc, apply=True)
    assert_old_builds_remain(deployment)


@pytest.mark.parametrize("escape", ["storage", "active", "rollback", "candidate"])
def test_release_directory_symlinks_cannot_escape_storage(deployment, escape):
    base, _, _ = deployment
    relative = "releases" if escape == "storage" else f"releases/{dict(active=ACTIVE, rollback=ROLLBACK, candidate=OLD)[escape]}"
    path = base / relative
    destination = base.parent / "outside"
    path.rename(destination)
    path.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError):
        prune(deployment, apply=True)
    assert destination.is_dir()
    assert_old_builds_remain(deployment)


def test_rechecks_active_release_immediately_before_removing_any_build(deployment, monkeypatch):
    base, _, _ = deployment
    original = cleanup.retained_releases
    checks = 0

    def change_current_after_planning(*args):
        nonlocal checks
        checks += 1
        if checks == 2:
            (base / "current").unlink()
            (base / "current").symlink_to(base / "releases" / ROLLBACK)
        return original(*args)

    monkeypatch.setattr(cleanup, "retained_releases", change_current_after_planning)
    with pytest.raises(ValueError, match="Active release changed"):
        prune(deployment, apply=True)
    assert checks == 2
    assert_old_builds_remain(deployment)


@pytest.mark.parametrize("hazard", ["mount", "cwd", "command", "dependency", "feedback"])
def test_used_or_unshared_builds_abort_the_entire_cleanup(deployment, hazard):
    base, _, proc = deployment
    old = base / "releases" / OLD
    if hazard == "mount":
        mounted = old / "mounted data"
        mounted.mkdir()
        encoded = str(mounted).replace(" ", r"\040")
        (proc / "self/mountinfo").write_text(f"10 9 8:1 / {encoded} rw - ext4 /dev/test rw\n")
    elif hazard in {"cwd", "command"}:
        process = proc / "4321"
        process.mkdir()
        (process / "cwd").symlink_to(old if hazard == "cwd" else base)
        command = [b"python", str(old / "gui/app.py").encode()] if hazard == "command" else [b"python"]
        (process / "cmdline").write_bytes(b"\0".join(command) + b"\0")
    elif hazard == "dependency":
        (base / "releases" / ROLLBACK / "models").symlink_to(old / "gui")
    else:
        (old / "feedback").unlink()
        (old / "feedback").mkdir()
        (old / "feedback/learner.json").write_text("Private learner data")
    with pytest.raises(ValueError):
        prune(deployment, apply=True)
    assert_old_builds_remain(deployment)


def test_a_process_that_has_exited_does_not_prevent_cleanup(deployment):
    _, _, proc = deployment
    process = proc / "4321"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0")
    assert prune(deployment, apply=True)["releases"] == [OLD, UNUSED]


def test_cli_refuses_cleanup_while_another_deployment_holds_the_lock(deployment):
    base, _, _ = deployment
    with (base / ".release.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/prune_releases.py"), "--base", str(base),
             "--active", ACTIVE, "--rollback", ROLLBACK, "--apply"],
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode != 0
    assert "Release cleanup stopped" in result.stderr
    assert_old_builds_remain(deployment)


def test_cli_reuses_inherited_deployment_lock_without_releasing_parents_lock(deployment):
    base, _, _ = deployment
    for name in (OLD, UNUSED):
        shutil.rmtree(base / "releases" / name)
    command = [sys.executable, str(ROOT / "tools/prune_releases.py"), "--base", str(base),
               "--active", ACTIVE, "--rollback", ROLLBACK]
    with (base / ".release.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = subprocess.run(
            [*command, "--lock-fd", str(lock.fileno())], pass_fds=(lock.fileno(),),
            capture_output=True, text=True, timeout=10,
        )
        assert inherited.returncode == 0, inherited.stderr
        assert "Would remove 0 old releases" in inherited.stdout
        # Closing the child's duplicated descriptor must not unlock the
        # deployment that owns the original open file description.
        competing = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert competing.returncode != 0
        assert "Release cleanup stopped" in competing.stderr


@pytest.mark.parametrize("health_fails,cleanup_fails", [(False, False), (True, False), (False, True)])
def test_activation_runs_cleanup_only_after_health_and_never_rolls_back_for_cleanup_failure(
    tmp_path, health_fails, cleanup_fails,
):
    """Exercise the real Bash branches; all host-specific commands are harmless shims."""
    root = tmp_path.resolve()
    base = root / "ai-lab"
    before, stage = make_release(base, ROLLBACK), make_release(base, ACTIVE)
    (base / "current").symlink_to(before)
    (stage / "previous-release.txt").write_text(str(before) + "\n")
    log = root / "commands.jsonl"
    bindir = root / "bin"
    bindir.mkdir()
    shim = bindir / "shim"
    shim.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\\n")
if name == "readlink":
    print(os.path.realpath(args[-1]))
elif name == "mv":
    os.replace(args[-2], args[-1])
elif name in {"python", "python3"}:
    if args and args[0].endswith("prune_releases.py"):
        sys.exit(int(os.environ["FAIL_CLEANUP"]))
elif name == "curl" and args[-1].startswith("https:"):
    sys.exit(int(os.environ["FAIL_HEALTH"]))
''')
    shim.chmod(0o755)
    for name in ("flock", "readlink", "mv", "python3", "systemctl", "curl"):
        (bindir / name).symlink_to(shim)
    (stage / ".venv/bin/python").unlink()
    (stage / ".venv/bin/python").symlink_to(shim)
    monitor = root / "monitor.service"
    monitor.touch()
    source = (ROOT / "tools/release_update.sh").read_text()
    source = source.replace("BASE=/opt/ai-lab", f"BASE={shlex.quote(str(base))}")
    source = source.replace("/etc/systemd/system/ai-lab-monitor.service", str(monitor))
    script = root / "release_update.sh"
    script.write_text(source)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", COMMAND_LOG=str(log),
               FAIL_HEALTH=str(int(health_fails)), FAIL_CLEANUP=str(int(cleanup_fails)))
    result = subprocess.run(["bash", str(script), "activate", ACTIVE], env=env,
                            text=True, capture_output=True, timeout=10)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    cleanup_commands = [c for c in commands if len(c) > 1 and c[1].endswith("prune_releases.py")]
    if health_fails:
        assert result.returncode != 0
        assert (base / "current").resolve() == before
        assert not cleanup_commands
        assert "Rolling AI Lab back" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert (base / "current").resolve() == stage
        assert len(cleanup_commands) == 1
        assert cleanup_commands[0][-7:] == ["--active", ACTIVE, "--rollback", ROLLBACK, "--lock-fd", "9", "--apply"]
        cleanup_index = commands.index(cleanup_commands[0])
        public_health_index = next(i for i, c in enumerate(commands) if c[0] == "curl" and c[-1].startswith("https:"))
        baseline_index = next(i for i, c in enumerate(commands) if c[-1] == "--set-baseline")
        assert public_health_index < baseline_index < cleanup_index
        assert "Rolling AI Lab back" not in result.stderr
        assert ("cleanup needs attention" in result.stderr) == cleanup_fails
