#!/usr/bin/env python3
"""Visitor statistics from the nginx access log — no app-side tracking required.

Deliberately reads the web-server logs the site already produces, rather than adding
analytics to the app: nothing extra is collected, no cookies, no third party, and the
data never leaves your server. Run it over SSH:

    ssh root@your-server
    cd /opt/ai-lab/app
    python3 tools/visitor_report.py                 # last 7 days
    python3 tools/visitor_report.py --today
    python3 tools/visitor_report.py --month
    python3 tools/visitor_report.py --days 90
    python3 tools/visitor_report.py --today --ips   # per-IP breakdown
    python3 tools/visitor_report.py --hours         # hour-of-day histogram

How the numbers are derived (this is a Streamlit app, so it is not naive hit-counting):

  sessions   – requests to /_stcore/stream, the websocket Streamlit opens once per
               browser tab. The closest honest proxy for "someone actually used it".
  page loads – GET / (and /?…): first visits AND reloads.
  reloads    – page loads minus sessions, i.e. loads that did not open a new socket.
  visitors   – distinct client IPs (a household or office shares one, so this
               under-counts people and over-counts phones roaming between networks).
  bots       – filtered out of the headline numbers by User-Agent and reported apart.

Reads .log and rotated .log.1 / .log.*.gz automatically.

Privacy: IP addresses appear only with --ips, only in your terminal, from logs you
already hold. Use --anonymise to mask the last octet if you want to share output.
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

DEFAULT_LOGS = [
    "/var/log/nginx/access.log",
    "/var/log/nginx/ai-lab.access.log",
]

# nginx "combined": IP - user [10/Oct/2025:13:55:36 +0000] "GET / HTTP/1.1" 200 1234 "ref" "ua"
LINE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>\S+) (?P<path>\S*?) [^"]*" '
    r'(?P<status>\d{3}) (?P<bytes>\S+) "(?P<ref>[^"]*)" "(?P<ua>[^"]*)"'
)
BOT = re.compile(r"bot|crawler|spider|slurp|bingpreview|headless|curl|wget|python-requests|"
                 r"scrapy|facebookexternalhit|semrush|ahrefs|mj12|dotbot|petal|censys|zgrab",
                 re.I)


def open_any(p: pathlib.Path):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" \
        else open(p, "r", errors="replace")


def find_logs(explicit: str | None, domain: str | None):
    if explicit:
        base = pathlib.Path(explicit)
        return sorted(base.parent.glob(base.name + "*"))
    out = []
    for cand in DEFAULT_LOGS:
        base = pathlib.Path(cand)
        out += sorted(base.parent.glob(base.name + "*")) if base.parent.exists() else []
    return out


def parse(paths, start: date, domain: str | None):
    rows = []
    for p in paths:
        try:
            with open_any(p) as fh:
                for line in fh:
                    m = LINE.match(line)
                    if not m:
                        continue
                    try:
                        dt = datetime.strptime(m["ts"].split()[0], "%d/%b/%Y:%H:%M:%S")
                    except ValueError:
                        continue
                    if dt.date() < start:
                        continue
                    rows.append({
                        "ip": m["ip"], "dt": dt, "day": dt.date().isoformat(),
                        "hour": dt.hour, "path": m["path"], "status": m["status"],
                        "ref": m["ref"], "ua": m["ua"],
                        "bot": bool(BOT.search(m["ua"])) or m["ua"] in ("", "-"),
                    })
        except OSError as e:
            print(f"  (cannot read {p}: {e})", file=sys.stderr)
    return rows


def mask(ip: str, on: bool) -> str:
    if not on:
        return ip
    if ":" in ip:                       # IPv6
        return ":".join(ip.split(":")[:3]) + "::/48"
    parts = ip.split(".")
    return ".".join(parts[:3] + ["x"]) if len(parts) == 4 else ip


def bar(n: int, top: int, width: int = 34) -> str:
    return "█" * max(1, round(n / top * width)) if n and top else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Visitor stats from the nginx access log.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--today", action="store_true")
    g.add_argument("--week", action="store_true", help="last 7 days (default)")
    g.add_argument("--month", action="store_true", help="last 30 days")
    g.add_argument("--days", type=int, metavar="N")
    ap.add_argument("--log", help="path to the access log (globs rotations too)")
    ap.add_argument("--domain", help="only count requests for this vhost (if logged)")
    ap.add_argument("--ips", action="store_true", help="show the per-IP breakdown")
    ap.add_argument("--hours", action="store_true", help="show the hour-of-day histogram")
    ap.add_argument("--pages", action="store_true", help="show the most-viewed pages")
    ap.add_argument("--anonymise", action="store_true", help="mask the last octet of IPs")
    ap.add_argument("--bots", action="store_true", help="include bots in the headline numbers")
    a = ap.parse_args()

    today = date.today()
    if a.today:
        start, label = today, "today"
    elif a.month:
        start, label = today - timedelta(days=30), "last 30 days"
    elif a.days:
        start, label = today - timedelta(days=a.days), f"last {a.days} days"
    else:
        start, label = today - timedelta(days=7), "last 7 days"

    paths = find_logs(a.log, a.domain)
    if not paths:
        print("No nginx access log found. Pass --log /path/to/access.log")
        return 1

    rows = parse(paths, start, a.domain)
    print(f"AI-LAB visitors — {label}")
    print(f"logs: {', '.join(str(p) for p in paths)}")
    print("=" * 66)
    if not rows:
        print("No requests in this window.")
        return 0

    bots = [r for r in rows if r["bot"]]
    human = rows if a.bots else [r for r in rows if not r["bot"]]
    if not human:
        print(f"Only bot traffic in this window ({len(bots)} requests).")
        return 0

    def is_load(r):    # the HTML document itself: a first visit or a reload
        return r["path"] in ("/", "") or r["path"].startswith("/?")

    def is_socket(r):  # Streamlit opens one websocket per browser tab
        return "_stcore/stream" in r["path"] or "/stream" == r["path"]

    loads = [r for r in human if is_load(r)]
    socks = [r for r in human if is_socket(r)]
    visitors = {r["ip"] for r in human}
    reloads = max(len(loads) - len(socks), 0)

    print(f"  visitors (unique IPs) : {len(visitors)}")
    print(f"  sessions (app opens)  : {len(socks)}")
    print(f"  page loads            : {len(loads)}   (of which reloads ≈ {reloads})")
    print(f"  total requests        : {len(human)}")
    print(f"  bot requests filtered : {len(bots)}")
    if visitors:
        print(f"  sessions per visitor  : {len(socks)/len(visitors):.1f}")

    # --- per day ---
    by_day_v = defaultdict(set)
    by_day_s = Counter()
    for r in human:
        by_day_v[r["day"]].add(r["ip"])
        if is_socket(r):
            by_day_s[r["day"]] += 1
    days = sorted(by_day_v)
    top = max((len(by_day_v[d]) for d in days), default=1)
    print(f"\n  {'day':<12} {'visitors':>8} {'sessions':>9}")
    for d in days:
        print(f"  {d:<12} {len(by_day_v[d]):>8} {by_day_s[d]:>9}  {bar(len(by_day_v[d]), top)}")

    # --- returning visitors ---
    seen_days = defaultdict(set)
    for r in human:
        seen_days[r["ip"]].add(r["day"])
    returning = sum(1 for ip, ds in seen_days.items() if len(ds) > 1)
    if len(days) > 1:
        print(f"\n  returning visitors (seen on >1 day): {returning} of {len(visitors)}")

    # --- referrers ---
    refs = Counter(r["ref"] for r in human
                   if r["ref"] not in ("-", "") and "ai-lab.gokcol.online" not in r["ref"])
    if refs:
        print("\n  top referrers:")
        for ref, n in refs.most_common(8):
            print(f"    {n:>5}  {ref[:70]}")

    if a.pages:
        pages = Counter(r["path"].split("?")[0] for r in human
                        if not r["path"].startswith(("/_stcore", "/static", "/media",
                                                     "/favicon", "/healthz", "/vendor")))
        print("\n  most requested paths:")
        for pth, n in pages.most_common(12):
            print(f"    {n:>5}  {pth[:60]}")

    if a.hours:
        hrs = Counter(r["hour"] for r in human if is_socket(r)) or Counter(r["hour"] for r in human)
        hi = max(hrs.values())
        print("\n  by hour (UTC):")
        for h in range(24):
            print(f"    {h:02d}:00 {hrs.get(h,0):>5}  {bar(hrs.get(h,0), hi, 28)}")

    if a.ips:
        per = Counter(r["ip"] for r in human)
        print(f"\n  per visitor (top 20 of {len(per)}):")
        print(f"    {'ip':<40} {'reqs':>6} {'sessions':>9}  days")
        socks_by_ip = Counter(r["ip"] for r in socks)
        for ip, n in per.most_common(20):
            print(f"    {mask(ip, a.anonymise):<40} {n:>6} {socks_by_ip.get(ip,0):>9}"
                  f"  {len(seen_days[ip])}")
        heavy = [(ip, n) for ip, n in per.most_common(5) if n > 500]
        if heavy:
            print("\n  ⚠ very heavy clients (possible scraping/abuse):")
            for ip, n in heavy:
                print(f"    {mask(ip, a.anonymise)}: {n} requests")

    # --- errors ---
    errs = Counter(r["status"] for r in human if r["status"][0] in "45")
    if errs:
        print("\n  error responses: " + ", ".join(f"{s}×{n}" for s, n in errs.most_common(6)))

    print("\n  Note: 'visitors' counts IPs, not people — shared networks under-count and "
          "\n  mobile roaming over-counts. Treat these as trends, not analytics-grade truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
