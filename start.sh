#!/usr/bin/env bash
# Start the AI Lab GUI (Streamlit).
#   ./start.sh [port]      (default 8501, or set AILAB_PORT)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${AILAB_PORT:-${1:-8501}}"
VENV="$HERE/.venv"
PIDFILE="$HERE/.gui.pid"
LOGFILE="$HERE/.gui.log"

# Already running (via our pidfile)?
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "AI Lab GUI already running (PID $(cat "$PIDFILE")) -> http://localhost:$PORT"
  echo "Use ./stop.sh first if you want to restart."
  exit 0
fi
rm -f "$PIDFILE"

# Port already taken by something else?
if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is in use. Run ./stop.sh, or pick another: ./start.sh <port>" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "No venv at $VENV" >&2
  echo "Create it:  python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[gui]'" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Local runs get the 🐍 Sandbox (arbitrary-code scratchpad); public deploys don't set
# this, so the Sandbox is hidden there.
# Sandbox = in-app arbitrary-Python page. Default ON for LOCAL use only.
# On a public server run:  AILAB_ENABLE_SANDBOX=0 ./start.sh   (never expose it).
export AILAB_ENABLE_SANDBOX="${AILAB_ENABLE_SANDBOX:-1}"

echo "Starting AI Lab GUI on port $PORT ..."
nohup streamlit run gui/app.py \
  --server.headless true \
  --server.port "$PORT" \
  --browser.gatherUsageStats false \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

# Wait for it to become healthy.
for _ in $(seq 1 30); do
  if curl -fs "http://localhost:$PORT/_stcore/health" >/dev/null 2>&1; then
    echo "Ready ->  http://localhost:$PORT   (PID $(cat "$PIDFILE"), logs: .gui.log)"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Server exited during startup. Last log lines:" >&2
    tail -n 20 "$LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 0.5
done

echo "Started (PID $(cat "$PIDFILE")) but health check timed out - check .gui.log" >&2
exit 1
