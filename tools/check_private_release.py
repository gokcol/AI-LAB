"""Reject private-only deployments that change any public source or release history."""
from pathlib import Path
import sys

PRIVATE_FILES = {'tools/release_update.sh', 'tools/check_private_release.py',
                 'tools/collect_ops_security.py', 'tools/install_ops_monitor.sh',
                 'tests/test_ops_dashboard.py', 'tests/test_ops_security.py',
                 'tools/visitor_report.py', 'ops/ai-lab-history.logrotate',
                 'docs/SECURITY-OPERATIONS.md'}


def manifest(root):
    rows = {}
    for line in (Path(root) / 'source-manifest.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in rows or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid source manifest')
        rows[name] = digest
    return rows


def validate(previous, stage):
    old, new = manifest(previous), manifest(stage)
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    forbidden = sorted(name for name in changed if not name.startswith('ops_dashboard/') and name not in PRIVATE_FILES)
    if forbidden:
        raise ValueError('Private-only release changes public source: ' + ', '.join(forbidden))
    if not changed:
        raise ValueError('No private changes to release')
    return changed


if __name__ == '__main__':
    try:
        changed = validate(*sys.argv[1:])
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    print(f'Private-only release: {len(changed)} source changes; public files unchanged')
