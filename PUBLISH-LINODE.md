# Publishing AI Lab on your Linode — step by step

Your setup: **Ubuntu on Linode, `139.162.159.186`, root SSH, other sites already running,
target `ai-lab.gokcol.online`.**

Because that box already serves other sites, everything below is written to **add** a vhost
without touching what is there. Nothing deletes, disables or rewrites another site's config.

Total time: about 15 minutes, most of it waiting for DNS.

---

## Step 0 · DNS first (do this now, then make coffee)

Certbot can only issue a certificate once the name resolves to the server, so start here.

At whoever runs DNS for `gokcol.online`, add:

| Type | Host / Name | Value | TTL |
|---|---|---|---|
| `A` | `ai-lab` | `139.162.159.186` | 300 |

Then check from your Mac until it answers correctly:

```bash
dig +short ai-lab.gokcol.online
```

You want exactly `139.162.159.186`. Empty or wrong → wait and retry; do not go on to Step 3
until this is right, or certbot will fail and you will hit Let's Encrypt's rate limit.

---

## Step 1 · Look before you touch (the box is not empty)

```bash
ssh root@139.162.159.186
```

```bash
nginx -v; ls /etc/nginx/sites-enabled/; ss -tlnp | grep -E ':(80|443|8501)'; df -h /; free -m
```

What you are checking:

- **`sites-enabled`** — the other sites. `ai-lab` must not already be there.
- **port 8501 free** — the app will bind it on localhost. If something else has it, run the
  deploy with `PORT=8502`.
- **disk and RAM** — the venv plus PyTorch-free requirements need ~400 MB; the service is
  capped at 2 GB RAM. On a 1 GB Nanode add swap first (Step 6).

---

## Step 2 · Find your region (it goes in the privacy notice)

The site tells visitors where their data is held, so the value has to be true. From the
**Linode dashboard → your Linode → Region** (e.g. *Frankfurt, DE*, *London, UK*).

I could not determine this from the IP: `139.162.159.186` is registered in the RIPE
(European) region, which narrows it to Europe but not to a city. Until you set it, the
notice says only "the European Union" (and "Avrupa Birliği" in the Turkish version).

The Turkish notice renders the country name itself: `REGION="Frankfurt, Germany"` shows as
*Frankfurt, Almanya*. If you pick a region the map does not cover, set
`AILAB_DATA_REGION_TR="..."` alongside it.

---

## Step 3 · Deploy (one command)

Still as root on the server:

```bash
wget -qO deploy.sh https://raw.githubusercontent.com/gokcol/AI-LAB/main/tools/deploy_ubuntu.sh
```

```bash
chmod +x deploy.sh && DOMAIN=ai-lab.gokcol.online REGION="Frankfurt, Germany" ./deploy.sh
```

Replace `REGION` with what Step 2 told you. No `EMAIL=` — you asked not to give one, and the
script handles that: it reuses this server's existing Let's Encrypt account (it has one if
any other site here is on HTTPS), and otherwise registers with
`--register-unsafely-without-email`. The only thing you lose is expiry-warning mail, and
renewal is automatic anyway.

The script is **idempotent** — safe to re-run any time. It:

1. installs `python3-venv`, `nginx`, `certbot` (skips whatever is present);
2. creates the unprivileged `ailab` system user — **the app never runs as root**;
3. clones/updates the repo into `/opt/ai-lab/app` and builds the venv;
4. writes the hardened `ai-lab.service` (see below);
5. writes **only** `/etc/nginx/sites-available/ai-lab` and symlinks it — your other vhosts
   are listed and left alone;
6. runs `nginx -t` before reloading, so a bad config can never take your other sites down;
7. obtains the certificate and turns on the HTTP→HTTPS redirect;
8. prints a verification summary.

The firewall is **opt-in** (`SETUP_FIREWALL=1`) precisely because enabling `ufw` blind on a
shared box can lock out your other services — see Step 6.

---

## Step 4 · Verify (from your Mac)

```bash
curl -sI https://ai-lab.gokcol.online | head -3
```


Then, **on the server**, the check that actually matters:

```bash
systemctl show ai-lab -p Environment --value | tr ' ' '\n' | grep -E 'AILAB_'
```

Expect exactly `AILAB_ENABLE_SANDBOX=0` plus your `AILAB_DATA_REGION`, and **no**
`AILAB_FEEDBACK_ADMIN` (that one would publish the feedback inbox). The Sandbox executes
visitor-supplied Python in the app's own process — on a public server that is a
remote-code-execution endpoint, and the app only registers the page when the value is
exactly `"1"`. Never export it in a shell profile on this machine.

> ⚠️ Do **not** verify this with `curl … | grep -i sandbox`. Streamlit serves a static
> shell and streams the page over a websocket, so that grep prints `0` **even when the
> Sandbox is fully enabled** — a false all-clear. I tested it. Check the unit
> environment above, then confirm by eye that the sidebar has no **Tools → Sandbox**.

Also check the service itself:

```bash
systemctl status ai-lab --no-pager; ss -tlnp | grep 8501
```

- status **active (running)**
- 8501 bound to **`127.0.0.1`** only — never `0.0.0.0`

Then in a browser: open the site, send yourself one test feedback, and confirm it arrives:

```bash
cd /opt/ai-lab/app && python3 tools/feedback_report.py --today
```

Finally confirm your other sites still work — `nginx -t` passed, but look anyway.

---

## Step 5 · Day-to-day

**Update to the latest commit** (re-running the deploy script does this too):

```bash
cd /opt/ai-lab/app && sudo -u ailab git pull && sudo -u ailab .venv/bin/pip install -q -r requirements.txt && systemctl restart ai-lab
```

**Read feedback** — there is deliberately no web admin page:

```bash
cd /opt/ai-lab/app && python3 tools/feedback_report.py --week
```

**Visitor stats**, straight from nginx's own log (nothing extra is collected):

```bash
cd /opt/ai-lab/app && python3 tools/visitor_report.py --week --ips
```

**Logs when something is wrong:**

```bash
journalctl -u ai-lab -n 100 --no-pager
```

**Weekly digests by cron** (`crontab -e`):

```cron
0 8 * * 1 cd /opt/ai-lab/app && python3 tools/feedback_report.py --week | mail -s "AI-LAB feedback" you@example.com
0 8 * * 1 cd /opt/ai-lab/app && python3 tools/visitor_report.py --week | mail -s "AI-LAB visitors" you@example.com
```

**Back up the one irreplaceable file** — everything else is in git:

```cron
30 3 * * * mkdir -p /opt/ai-lab/backup && cp /opt/ai-lab/app/feedback/feedback.jsonl /opt/ai-lab/backup/feedback-$(date +\%F).jsonl
```

---

## Step 6 · Optional hardening

**Swap**, if this is a 1 GB Nanode (pip builds can OOM without it):

```bash
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

**Firewall.** Only if `ufw` is currently inactive *and* you know every port your other
services need — allow SSH **first**, or you will lock yourself out:

```bash
ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw status
```

Add any other ports your existing sites use, and only then `ufw enable`.

**fail2ban** for SSH brute-force:

```bash
apt install -y fail2ban
```

**Cloudflare** in front if you ever want real DDoS protection. The app- and nginx-level
limits already in place stop abuse and scraping; they cannot stop a botnet saturating your
uplink — nothing running *on* the server can.

---

## Step 7 · If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| certbot: "challenge failed" | DNS not propagated | redo Step 0, wait, re-run the script |
| 502 Bad Gateway | app not running | `journalctl -u ai-lab -n 50` |
| 502 and app *is* running | port mismatch | `ss -tlnp \| grep 8501`; redeploy with `PORT=` |
| Site loads, others broke | nginx conflict | `nginx -t`, check for a duplicate `default_server` |
| Sandbox visible | env var leaked | `systemctl show ai-lab -p Environment`, fix unit, restart |
| Feedback not saving | permissions | `ls -la /opt/ai-lab/app/feedback` must be owned by `ailab` |
| Slow / OOM-killed | RAM | add swap (Step 6); `MemoryMax=2G` is in the unit |

**Full rollback** — removes only this site:

```bash
systemctl disable --now ai-lab && rm /etc/nginx/sites-enabled/ai-lab && nginx -t && systemctl reload nginx
```

Your other sites are untouched by that. `/opt/ai-lab` stays on disk (including feedback)
until you delete it yourself.

---

## What the server ends up running

```
visitor ──HTTPS──▶ nginx :443            (TLS, interactive rate limit, security headers)
                     │  proxy_pass + WebSocket upgrade
                     ▼
                  streamlit :8501        (127.0.0.1 only — unreachable from outside)
                  user: ailab            (unprivileged, no shell, no sudo)
                  NoNewPrivileges, ProtectSystem=strict, ProtectHome
                  writable: /opt/ai-lab/app/feedback  ← and nothing else
                  MemoryMax=2G, CPUQuota=150%
                  AILAB_ENABLE_SANDBOX=0 ← no code execution, ever
```

Even a total compromise of the Streamlit process gets an unprivileged account with one
writable directory, a read-only filesystem, no home directory and no privilege escalation.
