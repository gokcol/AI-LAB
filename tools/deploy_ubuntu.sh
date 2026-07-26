#!/usr/bin/env bash
# One-shot, idempotent deployment of AI Lab on Ubuntu (Linode).
# Run as root on the server:
#
#   wget -qO deploy.sh https://raw.githubusercontent.com/gokcol/AI-LAB/main/tools/deploy_ubuntu.sh
#   chmod +x deploy.sh
#   DOMAIN=ai-lab.gokcol.online REGION="Frankfurt, Germany" ./deploy.sh
#
# EMAIL is optional: if this server already has a Let's Encrypt account (it does if any
# other site here uses HTTPS) certbot reuses it. Set EMAIL=you@example.com only if you
# want expiry-warning mail.
#
# Safe to re-run: it updates the code, refreshes configs, and restarts cleanly.
# Point the DNS A/AAAA record at this server BEFORE running (certbot needs it).

set -euo pipefail

DOMAIN="${DOMAIN:-ai-lab.gokcol.online}"
EMAIL="${EMAIL:-}"                       # OPTIONAL — see the note above
REPO="${REPO:-https://github.com/gokcol/AI-LAB.git}"
APP_USER="${APP_USER:-ailab}"
APP_HOME="/opt/ai-lab"
APP_DIR="$APP_HOME/app"
PORT="${PORT:-8501}"
# Printed verbatim in the site's privacy notice, so make it true: the Linode region
# this server actually sits in. Find it in the Linode dashboard (Linodes -> Region).
REGION="${REGION:-the European Union}"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "Run as root (sudo -i)."

# --------------------------------------------------------------------------- #
say "1/8  Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git nginx ufw curl \
    certbot python3-certbot-nginx >/dev/null
ok "python3 $(python3 -V | cut -d' ' -f2), nginx, certbot, ufw"

PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 - <<'PY' || die "Python 3.10+ required (found $PYV)."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
ok "Python $PYV meets the 3.10+ requirement"

# --------------------------------------------------------------------------- #
say "2/8  Service user and code"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
    adduser --system --group --home "$APP_HOME" "$APP_USER" >/dev/null
    ok "created unprivileged user '$APP_USER'"
else
    ok "user '$APP_USER' exists"
fi
mkdir -p "$APP_HOME"
chown "$APP_USER:$APP_USER" "$APP_HOME"

if [ -d "$APP_DIR/.git" ]; then
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch --quiet origin
    sudo -u "$APP_USER" git -C "$APP_DIR" reset --hard --quiet origin/main
    ok "updated existing checkout"
else
    sudo -u "$APP_USER" git clone --quiet "$REPO" "$APP_DIR"
    ok "cloned $REPO"
fi
sudo -u "$APP_USER" mkdir -p "$APP_DIR/feedback"      # persistent feedback storage

# --------------------------------------------------------------------------- #
say "3/8  Python environment"
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
fi
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
ok "dependencies installed (torch is intentionally NOT installed — not needed by the site)"

# --------------------------------------------------------------------------- #
say "4/8  systemd service (sandbox DISABLED, admin inbox DISABLED)"
cat > /etc/systemd/system/ai-lab.service <<UNIT
[Unit]
Description=AI Lab (Streamlit)
After=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=MPLBACKEND=Agg
# CRITICAL: the Sandbox runs arbitrary visitor Python. Never enable it here.
Environment=AILAB_ENABLE_SANDBOX=0
# Shown in the privacy notice as where visitors' data is held.
Environment=AILAB_DATA_REGION=$REGION
# AILAB_FEEDBACK_ADMIN is deliberately absent: the inbox must not be public.
ExecStart=$APP_DIR/.venv/bin/streamlit run gui/app.py \\
    --server.address 127.0.0.1 \\
    --server.port $PORT \\
    --server.headless true \\
    --browser.gatherUsageStats false \\
    --server.enableCORS false \\
    --server.enableXsrfProtection true \\
    --server.maxUploadSize 1 \\
    --client.showErrorDetails none
Restart=always
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR/feedback
MemoryMax=2G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --quiet ai-lab
systemctl restart ai-lab
sleep 4
systemctl is-active --quiet ai-lab || { journalctl -u ai-lab -n 30 --no-pager; die "service failed to start"; }
ok "ai-lab.service running on 127.0.0.1:$PORT"

# --------------------------------------------------------------------------- #
say "5/8  nginx reverse proxy + rate limiting"
cat > /etc/nginx/sites-available/ai-lab <<NGINX
limit_req_zone  \$binary_remote_addr zone=ailab:10m rate=30r/m;
limit_conn_zone \$binary_remote_addr zone=ailabconn:10m;

server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    limit_req  zone=ailab burst=60 nodelay;
    limit_conn ailabconn 10;
    client_max_body_size 1m;

    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    # The lab is presented as its own site. The framework's HTML shell ships a
    # placeholder <title> and icon that would flash in the browser tab before the app
    # boots and replaces them; rewrite the title and serve our own icon at the path the
    # shell asks for, so the very first paint is already the lab's.
    sub_filter '<title>Streamlit</title>' '<title>AI Lab</title>';
    sub_filter_once on;
    gzip on;
    gzip_min_length 1024;
    gzip_types text/html text/css application/javascript application/json image/svg+xml;

    location = /favicon.png {
        alias $APP_DIR/gui/assets/favicon.png;
        access_log off;
        expires 7d;
    }

    location / {
        # sub_filter cannot rewrite a compressed upstream response, so ask the app for
        # plain text and let nginx do the compressing instead (gzip on, above).
        proxy_set_header Accept-Encoding "";
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    \$http_upgrade;     # websockets: required
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       \$host;
        # the app reads the LAST X-Forwarded-For entry for per-client rate limits
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP  \$remote_addr;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/ai-lab /etc/nginx/sites-enabled/ai-lab
# NOTE: we deliberately do NOT touch other vhosts. This server may already host
# other sites; our server_name block matches only $DOMAIN.
OTHER=$(ls /etc/nginx/sites-enabled/ 2>/dev/null | grep -v '^ai-lab$' | tr '\n' ' ')
[ -n "$OTHER" ] && warn "other nginx sites present (left untouched): $OTHER"
nginx -t >/dev/null 2>&1 || { nginx -t; die "nginx config invalid"; }
systemctl reload nginx
ok "nginx proxying $DOMAIN -> 127.0.0.1:$PORT"

# --------------------------------------------------------------------------- #
say "6/8  Firewall (opt-in)"
if [ "${SETUP_FIREWALL:-0}" = "1" ]; then
    ufw allow OpenSSH >/dev/null 2>&1 || true
    ufw allow 'Nginx Full' >/dev/null 2>&1 || true
    ufw --force enable >/dev/null 2>&1 || true
    ok "ufw enabled: SSH + HTTP/HTTPS"
else
    warn "skipped (this server may run other services). To enable later:"
    warn "  ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw enable"
    ok "the app stays private regardless — it binds 127.0.0.1 only"
fi

# --------------------------------------------------------------------------- #
say "7/8  HTTPS (Let's Encrypt)"
RESOLVED=$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1 || true)
MYIP=$(curl -fsS4 https://icanhazip.com 2>/dev/null || echo "")
if [ -z "$RESOLVED" ]; then
    warn "$DOMAIN does not resolve yet — skipping certbot. Set the DNS A record, then run:"
    warn "  certbot --nginx -d $DOMAIN --redirect"
elif [ -n "$MYIP" ] && [ "$RESOLVED" != "$MYIP" ]; then
    warn "$DOMAIN resolves to $RESOLVED but this host is $MYIP — skipping certbot."
    warn "  Fix DNS, then run: certbot --nginx -d $DOMAIN --redirect"
else
    # Email is OPTIONAL. Three cases, in order of preference:
    #   1. certbot already has an ACME account here (very likely if other sites use TLS)
    #      -> reuse it, no email needed at all
    #   2. EMAIL was provided -> register with it (gets expiry-warning mail)
    #   3. neither -> register without an email (renewal still works; no warning mail)
    if [ -d /etc/letsencrypt/accounts ] && [ -n "$(ls -A /etc/letsencrypt/accounts 2>/dev/null)" ]; then
        ACME_ARGS=""
        ok "existing Let's Encrypt account found — reusing it (no email required)"
    elif [ -n "$EMAIL" ]; then
        ACME_ARGS="-m $EMAIL"
    else
        ACME_ARGS="--register-unsafely-without-email"
        warn "no email given — registering without one (renewal still automatic,"
        warn "  but you will get no expiry-warning mail if renewal ever breaks)"
    fi
    # shellcheck disable=SC2086
    if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos $ACME_ARGS --redirect; then
        ok "HTTPS enabled (auto-renewal via the certbot systemd timer)"
        systemctl list-timers 2>/dev/null | grep -q certbot \
            && ok "renewal timer active" \
            || warn "check the renewal timer: systemctl status certbot.timer"
    else
        warn "certbot failed — run it manually:"
        warn "  certbot --nginx -d $DOMAIN --redirect"
    fi
fi

# --------------------------------------------------------------------------- #
say "8/8  Verification"
FAIL=0
if curl -fsS "http://127.0.0.1:$PORT" >/dev/null 2>&1; then ok "app answers locally"
else warn "app did not answer on 127.0.0.1:$PORT"; FAIL=1; fi

if ss -tlnp 2>/dev/null | grep -q "127.0.0.1:$PORT"; then
    ok "port $PORT bound to localhost only (not world-reachable)"
else
    warn "port $PORT is NOT bound to 127.0.0.1 — check the unit file"; FAIL=1
fi

if systemctl show ai-lab -p Environment | grep -q 'AILAB_ENABLE_SANDBOX=0'; then
    ok "Sandbox DISABLED (no remote-code-execution page)"
else
    warn "Sandbox flag not 0 — FIX THIS BEFORE GOING PUBLIC"; FAIL=1
fi

if systemctl show ai-lab -p Environment | grep -q 'AILAB_FEEDBACK_ADMIN'; then
    warn "AILAB_FEEDBACK_ADMIN is set — the feedback inbox would be PUBLIC. Remove it."; FAIL=1
else
    ok "feedback inbox is not exposed"
fi

BODY=$(curl -fsS "http://127.0.0.1:$PORT" 2>/dev/null || true)
if printf '%s' "$BODY" | grep -qi sandbox; then
    warn "the served page mentions 'sandbox' — investigate"; FAIL=1
else
    ok "no sandbox reference in the served page"
fi

printf '\n'
if [ "$FAIL" -eq 0 ]; then
    printf '\033[1;32m✅ Deployed.\033[0m  https://%s\n\n' "$DOMAIN"
else
    printf '\033[1;33m⚠  Deployed with warnings — review the ! lines above.\033[0m\n\n'
fi
cat <<NEXT
Useful commands
  systemctl status ai-lab            # service state
  journalctl -u ai-lab -f            # live logs
  systemctl restart ai-lab           # restart

Read visitor feedback (never exposed on the web)
  cd $APP_DIR && python3 tools/feedback_report.py            # last 7 days
  cd $APP_DIR && python3 tools/feedback_report.py --today
  cd $APP_DIR && python3 tools/feedback_report.py --month

Update to the latest version
  DOMAIN=$DOMAIN ./deploy.sh         # re-run this script
NEXT
