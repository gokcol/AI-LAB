# Public About statistics

The public app reads `/opt/ai-lab/shared/public-stats.json` (override with
`AILAB_PUBLIC_STATS_FILE`) and never needs log permissions. The snapshot contains
dates and counts only. Nothing from the private console is published.

Visits mean successful GETs (200 or 304) to registered page paths. Reloads count;
navigation within an existing Streamlit tab may not. Recognizable bots, probes,
loopback and the server's own IP are excluded. Optional `AILAB_OWN_IPS` in
`/etc/ai-lab-public-stats.env` adds comma- or space-separated exclusions.
These counts are not distinct people or analytics sessions, and bot detection is
heuristic. This definition is also displayed in About.

The collector reads retained nginx logs using the existing server's private
`tools/visitor_report.py` parser. It publishes completed UTC days starting
2026-10-01, atomically, and keeps previously published daily counts when logs expire.
An unreadable log aborts refresh, preserving the last snapshot. Malformed or
non-HTTP request lines are skipped; only parseable successful page loads count.
Days with no retained records and no prior aggregate are unavailable, not zero.
A day with log records but no qualifying page loads is a recorded zero. Log
presence cannot establish that every request in that day was captured.

Weekly bars run Monday–Sunday. Partial calendar weeks are labelled. Weeks with
missing days have no bar; their recorded subtotal and coverage remain in the table.
The daily average uses only recorded completed days, including recorded zero days.
An incomplete total is labelled as recorded visits. Snapshots older than two days
show a stale-data warning.

After activating a release containing this collector, install its units and run:

```sh
install -m 0644 /opt/ai-lab/current/ops/ai-lab-public-stats.service /etc/systemd/system/
install -m 0644 /opt/ai-lab/current/ops/ai-lab-public-stats.timer /etc/systemd/system/
systemctl daemon-reload
systemctl start ai-lab-public-stats.service
systemctl enable --now ai-lab-public-stats.timer
```

The timer runs at 00:15 UTC daily and catches up after downtime. It reads code
through the active-release symlink, so subsequent deployments use the current
collector. Back up the shared aggregate along with other shared data; it retains
counts beyond the one-year raw-log retention window. Do not commit snapshots.
