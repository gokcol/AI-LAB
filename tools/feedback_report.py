#!/usr/bin/env python3
"""Feedback report — the simple, secure way to read what visitors sent.

Runs LOCALLY on the server over SSH (stdlib only, read-only, nothing exposed to the
web — this is deliberately not an admin web page):

    ssh you@server
    cd ~/ai-lab
    python3 tools/feedback_report.py            # last 7 days (default)
    python3 tools/feedback_report.py --today
    python3 tools/feedback_report.py --month
    python3 tools/feedback_report.py --all --limit 100
    python3 tools/feedback_report.py --since 2026-07-01
    python3 tools/feedback_report.py --summary   # counts only, no messages

Automated digests (optional) — add a cron line on the server:

    # weekly email every Monday 08:00
    0 8 * * 1  cd /home/you/ai-lab && python3 tools/feedback_report.py --week | mail -s "AI-LAB feedback (weekly)" you@example.com
    # daily count-only ping
    0 8 * * *  cd /home/you/ai-lab && python3 tools/feedback_report.py --today --summary | mail -s "AI-LAB feedback (daily)" you@example.com

Messages are stored plain-text-validated and JSON-escaped by gui/feedback.py, and this
tool prints them verbatim to the terminal — nothing is interpreted or executed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from datetime import date, datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "feedback" / "feedback.jsonl"


def load(path: pathlib.Path):
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            # ts format: "2026-07-26 09:15:04 UTC"
            r["_day"] = str(r.get("ts", ""))[:10]
            out.append(r)
        except json.JSONDecodeError:
            print(f"  (skipping unparseable line {i})", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only feedback report (see module docstring).")
    period = ap.add_mutually_exclusive_group()
    period.add_argument("--today", action="store_true", help="today (UTC) only")
    period.add_argument("--week", action="store_true", help="last 7 days (default)")
    period.add_argument("--month", action="store_true", help="last 30 days")
    period.add_argument("--all", action="store_true", help="everything on file")
    period.add_argument("--since", metavar="YYYY-MM-DD", help="from this date")
    ap.add_argument("--summary", action="store_true", help="counts only, no message bodies")
    ap.add_argument("--limit", type=int, default=50, help="max messages to print (default 50)")
    ap.add_argument("--path", type=pathlib.Path, default=DEFAULT_FILE,
                    help=f"feedback file (default {DEFAULT_FILE})")
    a = ap.parse_args()

    today = date.today()
    if a.today:
        start, label = today, "today"
    elif a.month:
        start, label = today - timedelta(days=30), "last 30 days"
    elif a.all:
        start, label = date(1970, 1, 1), "all time"
    elif a.since:
        try:
            start = datetime.strptime(a.since, "%Y-%m-%d").date()
        except ValueError:
            print("--since must be YYYY-MM-DD"); return 2
        label = f"since {start}"
    else:
        start, label = today - timedelta(days=7), "last 7 days"

    rows = load(a.path)
    sel = [r for r in rows if r["_day"] >= start.isoformat()]

    print(f"AI-LAB feedback — {label}  ·  file: {a.path}")
    print(f"{'=' * 64}")
    print(f"entries: {len(sel)}   (all time: {len(rows)},"
          f" file: {a.path.stat().st_size:,} bytes)" if a.path.exists()
          else "no feedback file yet")
    if not sel:
        return 0

    by_day = Counter(r["_day"] for r in sel)
    by_client = Counter(r.get("client", "?") for r in sel)
    with_contact = sum(1 for r in sel if r.get("email"))
    avg_len = sum(len(r.get("message", "")) for r in sel) / len(sel)

    print(f"with contact email: {with_contact}   unique clients: {len(by_client)}   "
          f"avg length: {avg_len:.0f} chars")
    print("\nper day:")
    for d in sorted(by_day):
        print(f"  {d}  {'█' * min(by_day[d], 50)} {by_day[d]}")
    heavy = [(c, n) for c, n in by_client.most_common(3) if n >= 5]
    if heavy:
        print("\n⚠ heavy senders (possible abuse):")
        for c, n in heavy:
            print(f"  client {c}: {n} entries")

    if not a.summary:
        print(f"\nmessages (newest first, up to {a.limit}):")
        print("-" * 64)
        for r in sel[::-1][: a.limit]:
            who = " ".join(x for x in (r.get("name"), r.get("surname")) if x) or "anonymous"
            contact = f"  <{r['email']}>" if r.get("email") else ""
            print(f"[{r.get('ts', '?')}] {who}{contact}")
            print(f"    {r.get('message', '')}")
        if len(sel) > a.limit:
            print(f"... and {len(sel) - a.limit} more (use --limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
