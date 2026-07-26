# Deploying AI Lab on your own server (ai-lab.gokcol.online)

Ubuntu + nginx + systemd. Read the **Security** section first — especially the Sandbox.

---

## ⛔ 1. The Sandbox — never expose it

`gui/views/sandbox.py` executes **arbitrary Python typed by the visitor, in the app's own
process, with your user's permissions**. On a public server that is not "a bit risky" — it is
a **remote code execution endpoint**. A visitor could read `~/.ssh`, exfiltrate environment
variables and API keys, install a crypto-miner or reverse shell, delete files, or attack other
machines from your IP. The "Safe run" toggle only adds a subprocess timeout; it is **not** a
security boundary (no seccomp, no container, no filesystem isolation).

It exists because a *local* scratchpad is genuinely useful for learning. It is therefore
**off unless explicitly enabled**:

```bash
# app.py only registers the page when this is exactly "1"
SANDBOX_ENABLED = os.environ.get("AILAB_ENABLE_SANDBOX") == "1"
```

**Rules for the public server**

```bash
# start it with the sandbox explicitly disabled
AILAB_ENABLE_SANDBOX=0 ./start.sh
```

- The systemd unit below sets `Environment=AILAB_ENABLE_SANDBOX=0` — do not remove it.
- Never `export AILAB_ENABLE_SANDBOX=1` in a shell profile on that machine.
- Verify after every deploy: the sidebar must have **no "Tools → Sandbox"** entry.

```bash
# on the server — the only check that can actually see the setting
systemctl show ai-lab -p Environment --value | tr ' ' '\n' | grep -E 'AILAB_'
```

> ⚠️ Do **not** verify this with `curl … | grep -i sandbox`. Streamlit serves a static
> shell and streams the page over a websocket, so that grep prints `0` **even when the
> Sandbox is fully enabled** — a false all-clear. I tested it. Check the unit
> environment above, then confirm by eye that the sidebar has no **Tools → Sandbox**.

If you ever *do* want a public scratchpad, it must run in a locked-down container
(gVisor/Firejail, no network, read-only FS, CPU/memory caps) — a separate project, not a flag.

Same rule for the admin inbox: **`AILAB_FEEDBACK_ADMIN` must stay unset** on the server, or
the feedback list becomes public. Read feedback over SSH instead (§5).

---

## 2. Install

```bash
sudo adduser --system --group --home /opt/ai-lab ailab
sudo -u ailab git clone https://github.com/gokcol/AI-LAB.git /opt/ai-lab/app
cd /opt/ai-lab/app
sudo -u ailab python3 -m venv .venv
sudo -u ailab .venv/bin/pip install -r requirements.txt
sudo -u ailab mkdir -p /opt/ai-lab/app/feedback     # persistent feedback storage
```

Run as a **dedicated unprivileged user** — never root.

## 3. systemd service

`/etc/systemd/system/ai-lab.service`:

```ini
[Unit]
Description=AI Lab (Streamlit)
After=network.target

[Service]
Type=simple
User=ailab
Group=ailab
WorkingDirectory=/opt/ai-lab/app
Environment=MPLBACKEND=Agg
Environment=AILAB_ENABLE_SANDBOX=0
# AILAB_FEEDBACK_ADMIN is deliberately NOT set
ExecStart=/opt/ai-lab/app/.venv/bin/streamlit run gui/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --server.maxUploadSize 1
Restart=always
RestartSec=5

# hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/ai-lab/app/feedback
MemoryMax=2G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ai-lab
```

`--server.address 127.0.0.1` means Streamlit is reachable **only** through nginx.

## 4. nginx + TLS + rate limiting

`/etc/nginx/sites-available/ai-lab`:

```nginx
# rate-limit zones (http context)
limit_req_zone  $binary_remote_addr zone=ailab:10m rate=30r/m;
limit_conn_zone $binary_remote_addr zone=ailabconn:10m;

server {
    server_name ai-lab.gokcol.online;

    # app-wide limits: burst absorbs normal Streamlit chatter
    limit_req   zone=ailab burst=60 nodelay;
    limit_conn  ailabconn 10;
    client_max_body_size 1m;

    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    # Present the lab as its own site: rewrite the HTML shell's placeholder tab title
    # and serve our own icon at the path it requests, so neither flashes on first paint.
    sub_filter '<title>Streamlit</title>' '<title>AI Lab</title>';
    sub_filter_once on;
    gzip on;
    gzip_min_length 1024;
    gzip_types text/html text/css application/javascript application/json image/svg+xml;

    location = /favicon.png {
        alias /opt/ai-lab/app/gui/assets/favicon.png;
        access_log off;
        expires 7d;
    }

    location / {
        # sub_filter cannot rewrite a compressed upstream response, so take it plain
        # and let nginx compress instead.
        proxy_set_header Accept-Encoding "";
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;      # websockets: required
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       $host;
        # the app reads the LAST X-Forwarded-For entry for per-client limits
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP  $remote_addr;
        proxy_read_timeout 3600s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/ai-lab /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d ai-lab.gokcol.online --redirect   # HTTPS
# No email required if this server already has a Let's Encrypt account
# (it does if any other site here uses HTTPS). Otherwise certbot will ask,
# or you can skip it with: --register-unsafely-without-email
```

Firewall + brute-force protection:

```bash
sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable
sudo apt install fail2ban          # protects SSH; can also watch nginx logs
```

For real DDoS resilience (volumetric attacks), put **Cloudflare** in front — app- and
nginx-level limits stop abuse, not a botnet saturating your uplink.

## 5. Reading feedback (securely)

Feedback lands in `/opt/ai-lab/app/feedback/feedback.jsonl` and is **git-ignored**. There is
deliberately no public admin page — read it over SSH:

```bash
ssh you@server
cd /opt/ai-lab/app
python3 tools/feedback_report.py             # last 7 days
python3 tools/feedback_report.py --today
python3 tools/feedback_report.py --month
python3 tools/feedback_report.py --all --limit 100
python3 tools/feedback_report.py --today --summary    # counts only
```

Automatic digests by email (`sudo apt install mailutils`), as the `ailab` user:

```cron
# weekly digest, Mondays 08:00
0 8 * * 1 cd /opt/ai-lab/app && python3 tools/feedback_report.py --week | mail -s "AI-LAB feedback (weekly)" you@example.com
# daily one-liner
0 8 * * * cd /opt/ai-lab/app && python3 tools/feedback_report.py --today --summary | mail -s "AI-LAB feedback (daily)" you@example.com
```

Back it up (it is the only non-reproducible data on the box):

```cron
30 3 * * * cp /opt/ai-lab/app/feedback/feedback.jsonl /opt/ai-lab/backup/feedback-$(date +\%F).jsonl
```

## 5b. Visitor statistics (from the logs you already have)

No analytics were added to the app — the stats come from nginx's own access log, so
nothing extra is collected and no data leaves the box:

```bash
cd /opt/ai-lab/app
python3 tools/visitor_report.py                 # last 7 days
python3 tools/visitor_report.py --today
python3 tools/visitor_report.py --month
python3 tools/visitor_report.py --today --ips   # per-IP breakdown + heavy clients
python3 tools/visitor_report.py --hours --pages # when, and what they read
python3 tools/visitor_report.py --ips --anonymise   # masked IPs, safe to paste
```

Reported: unique visitors (IPs), **sessions** (Streamlit websocket opens — the honest
"someone used it" number), **page loads and reloads**, returning visitors, referrers,
top paths, hour-of-day, error responses, and bot traffic filtered out and counted apart.

Weekly digest by cron:

```cron
0 8 * * 1 cd /opt/ai-lab/app && python3 tools/visitor_report.py --week | mail -s "AI-LAB visitors (weekly)" you@example.com
```

Because the report reads the access log, the site's Terms say plainly that access logs
are kept and reviewed for security and basic visitor counts. Keep that accurate if you
change what you collect.

## 6. Application-level protections (already built in)

| layer | limit |
|---|---|
| message | plain text only; no HTML/markup/code/entities/links; **200 chars**; control chars rejected |
| name / surname / email | optional; letters-only names ≤ 50; validated email ≤ 100 |
| bot traps | hidden honeypot field + minimum time-to-fill — both **fail silently** |
| per session | 60 s cooldown, max 3 submissions |
| per client (IP) | 5 per hour |
| global | 30 per hour, **50 stored per day** |
| storage | file capped at 256 KB; entries JSON-escaped; rendered only via `st.text` |
| privacy | client IP is **hashed** (never stored raw) |

## 7. Updating

```bash
cd /opt/ai-lab/app
sudo -u ailab git pull
sudo -u ailab .venv/bin/pip install -r requirements.txt
sudo systemctl restart ai-lab
```

## 8. Post-deploy checklist

- [ ] `systemctl show ai-lab -p Environment` shows `AILAB_ENABLE_SANDBOX=0` and **no** `AILAB_FEEDBACK_ADMIN`
- [ ] `ss -tlnp | grep 8501` → bound to **127.0.0.1** only
- [ ] HTTPS valid; HTTP redirects
- [ ] submit test feedback → appears via `tools/feedback_report.py --today`
- [ ] `feedback/` is git-ignored (`git status` clean after a submission)
