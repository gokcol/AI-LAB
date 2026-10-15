"""Publish daily page-load counts only; retain counts after raw log rotation."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui.public_stats import START, validate

PAGE_PATHS = {"", "/"} | {"/"+p.stem for p in (Path(__file__).resolve().parents[1]/"gui/views").glob("*.py")}


def build_snapshot(rows, coverage, now, previous=None, own_ips=()):
    end = now.astimezone(timezone.utc).date()-timedelta(days=1)
    counts = Counter()
    for row in rows:
        path = unquote(unquote(row["path"].split("?", 1)[0])).rstrip("/")
        if (START.isoformat() <= row["day"] <= end.isoformat() and not row["bot"]
                and not row["probe"] and row["ip"] not in own_ips and row["method"] == "GET"
                and row["status"] in ("200", "304") and path in PAGE_PATHS):
            counts[row["day"]] += 1
    old = {r["day"]: r["visits"] for r in validate(previous)["daily"]} if previous else {}
    daily = []
    for offset in range((end-START).days+1):
        day = (START+timedelta(days=offset)).isoformat()
        # Rotated history can begin partway through a day. Never erase counts
        # already published; page-load totals are cumulative, not unique visitors.
        count = max(counts[day], old.get(day) or 0) if day in coverage.get("days", set()) else old.get(day)
        daily.append({"day": day, "visits": count})
    # Malformed/non-HTTP request lines are not successful page loads. The shared
    # parser skips these (including scanner garbage); I/O failures are different.
    if coverage.get("errors"):
        raise ValueError("Incomplete log read; retaining the previous public snapshot")
    return validate({"version": 1, "start": START.isoformat(), "through": end.isoformat(),
                     "updated": now.isoformat(), "daily": daily})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="/var/log/nginx/ai-lab.access.log")
    parser.add_argument("--output", type=Path, default=Path("/opt/ai-lab/shared/public-stats.json"))
    args = parser.parse_args()
    # The existing server supplies this private log parser. No private modules are
    # imported by the public app, and no identifying fields enter the snapshot.
    from tools import visitor_report
    now = datetime.now(timezone.utc)
    paths = visitor_report.find_logs(args.log, None)
    if not paths:
        raise SystemExit("No logs available; previous snapshot retained")
    coverage = {}
    rows = visitor_report.iter_rows(paths, START, now.date()-timedelta(days=1), coverage)
    previous = json.loads(args.output.read_text()) if args.output.exists() else None
    own_ips = set(os.environ.get("AILAB_OWN_IPS", "").replace(",", " ").split()) | {"127.0.0.1", "::1", "139.162.159.186"}
    snapshot = build_snapshot(rows, coverage, now, previous, own_ips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False, prefix="public-stats-") as file:
        json.dump(snapshot, file)
        temp = Path(file.name)
    os.chmod(temp, 0o644)
    temp.replace(args.output)
    print(f"Public statistics updated through {snapshot['through']}; aggregate counts only")


if __name__ == "__main__":
    main()
