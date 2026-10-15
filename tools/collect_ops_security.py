"""Collect sanitized ops data. Installed as a root-owned, scheduled service."""
import argparse
import grp
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops_dashboard.security import collect, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set-baseline', action='store_true', help='Approve current release after deployment verification')
    args = parser.parse_args()
    root = Path('/opt/ai-lab/current').resolve()
    directory = Path('/var/lib/ai-lab-monitor')
    directory.mkdir(mode=0o750, exist_ok=True)
    os.chown(directory, 0, grp.getgrnam('ailab-admin').gr_gid)
    baseline_file = directory / 'baseline.json'
    if args.set_baseline:
        baseline = {'release': str(root), 'manifest_sha256': digest(root / 'source-manifest.sha256')}
        fd = os.open(baseline_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as file:
            json.dump(baseline, file)
    try:
        baseline = json.loads(baseline_file.read_text())
    except (OSError, ValueError):
        baseline = {}
    data = collect(root, baseline)
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='status-', delete=False) as file:
        temporary = Path(file.name)
        json.dump(data, file)
    os.chmod(temporary, 0o640)
    os.chown(temporary, 0, grp.getgrnam('ailab-admin').gr_gid)
    temporary.replace(directory / 'status.json')
    print('Ops snapshot updated: ' + ', '.join(f'{k}={"available" if data[k]["available"] else "unavailable"}' for k in ('bans','integrity','errors')))

if __name__ == '__main__':
    main()
