# Public deployment protections

The public service sets `AILAB_RESOURCE_LIMITS=1`. A single Streamlit process admits
at most three simultaneous page computations, with no unbounded job queue. Each
session has 60 page executions/minute with a burst of 20; each client IP has 600/minute
with a burst of 120. This also limits reruns arriving over existing WebSockets.

Expensive cache misses and experiment runs share a training budget of 12/minute per
session (burst 2) and 120/minute per IP (burst 30). Nested calculations in one rerun
charge once. A retry notice appears when admission is refused. Cached results remain
available subject to the page budget. Client counters expire and their total is bounded.
The limiter is process-wide: multiple Streamlit worker processes would require a
shared counter service. Local CLI experiments do not use these public-server limits.

The existing-host nginx policy allows 120 HTTP requests/minute per IP, burst 120,
and 30 concurrent connections. Framework static assets are outside these limits.
HTTP rejections use 429 with Retry-After: 30. The normalized-path map in
`ops/nginx-ai-lab-probes.conf` rejects obvious sensitive-file and exploit probes before
they reach the application; ACME `.well-known` paths are excluded. Rejected probes
have a dedicated rotating log, with six probes in ten minutes resulting in a one-hour
temporary web-port ban. SSH failures have a separate jail.

Keep sandbox/tests/heavy jobs disabled in the public service, and retain its unprivileged
user, filesystem protection, 2 GB memory limit, and 150% CPU quota. BLAS thread counts
are pinned to one. These controls reduce abuse impact; they are not a substitute for
patching or protection against a large distributed flood.

Every public update must prepend a release entry to `gui/content/releases.json`.
Operational credentials and host recovery notes belong in private storage, not this
public repository or the public What's new section.

## Private diagnostics collector (2026-09-28)

`ai-lab-monitor.timer` runs a root-owned, read-only collector about every minute.
The web process has no new privileges and only reads
`/var/lib/ai-lab-monitor/status.json` (root:ailab-admin, 0640). The directory is
0750; `baseline.json` is root-only, 0600. No raw journal messages are stored.

Install with `bash tools/install_ops_monitor.sh`, after deploying source. Approve
the initial release only after verifying its manifest using
`.venv/bin/python tools/collect_ops_security.py --set-baseline`. Later validated
activations approve the new manifest automatically. Never rebaseline merely to
clear an integrity warning. Dependencies, operating-system files and runtime
data are outside the integrity scope. Check collection failures with
`journalctl -u ai-lab-monitor.service`; stale snapshots are visibly marked.

Private-only changes use `tools/release_update.sh stage|activate RELEASE
--private-ops`. Both steps validate public release notes and compare source
manifests, rejecting changes outside the explicit private-file allowlist. Public
release history remains unchanged; private changes belong in
`ops_dashboard/CHANGES.md`. Normal releases still require a new public entry.

Application failures are recognized from Streamlit's uncaught-execution and
ASGI error markers. Journal retention and the 10,000-entry cap limit coverage.
Events are not tied to a request status or visitor; HTTP 200 alone does not prove
lesson success. Handled errors and client-side failures need separate telemetry.

## Traffic history retention (2026-10-15)

The ops dashboard reconstructs traffic from AI Lab's dedicated nginx access log;
there is no independent historical analytics database. The former shared nginx
policy kept 14 rotations, so by October 15 only October 1 onward remained.
An older server/log backup is required to recover expired September traffic.

Install `ops/ai-lab-history.logrotate` as `/etc/logrotate.d/ai-lab-history`.
This file sorts before `nginx` and takes ownership of **only**
`/var/log/nginx/ai-lab.access.log`, keeping 366 rotations with a 366-day age cap.
`ignoreduplicates` (supported by the server's logrotate 3.21) prevents the generic
nginx wildcard from claiming it again. Other sites and probe logs keep their
existing policy. No paths change, so the dashboard reads older rotations normally.
The number of retained days grows from the earliest surviving log, not retroactively.

After installation run `logrotate --debug /etc/logrotate.conf` and verify the
AI Lab access log is listed under the 366-rotation rule exactly once, while other
nginx logs remain under 14 rotations. Do not force a production rotation just to
test it. Keep a root-only backup of the previous configuration when replacing it.

Daily report rows with no retained entries display Unavailable; CSV count cells
are blank. A recorded day with only bots or excluded health checks can correctly
show zero visitors. Retained entries do not prove the entire day was logged.
