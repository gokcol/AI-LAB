#!/usr/bin/env bash
# Install private read-only monitoring. Baseline approval is a separate deploy step.
set -euo pipefail
[[ $(id -u) == 0 ]] || exit 1
cat > /etc/systemd/system/ai-lab-monitor.service <<'UNIT'
[Unit]
Description=AI Lab read-only operations diagnostics collector
After=fail2ban.service
[Service]
Type=oneshot
ExecStart=/opt/ai-lab/current/.venv/bin/python /opt/ai-lab/current/tools/collect_ops_security.py
User=root
UMask=0027
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/ai-lab-monitor
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
MemoryMax=512M
CPUQuota=50%
TimeoutStartSec=110
UNIT
cat > /etc/systemd/system/ai-lab-monitor.timer <<'UNIT'
[Unit]
Description=Refresh AI Lab operations diagnostics every minute
[Timer]
OnBootSec=30
OnUnitInactiveSec=60
[Install]
WantedBy=timers.target
UNIT
install -d -m 0750 -o root -g ailab-admin /var/lib/ai-lab-monitor
systemctl daemon-reload
systemctl enable --now ai-lab-monitor.timer
