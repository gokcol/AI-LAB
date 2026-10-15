#!/usr/bin/env bash
# Stage an AI Lab release on the existing server. Does not change nginx or other apps.
set -Eeuo pipefail
MODE="${1:-}"
RELEASE="${2:-}"
BASE=/opt/ai-lab
[[ "$RELEASE" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo 'Expected release YYYYMMDDTHHMMSSZ'; exit 2; }
STAGE="$BASE/releases/$RELEASE"
PREVIOUS=$(readlink -f "$BASE/current")
[[ "$PREVIOUS" == "$BASE/releases/"* && -d "$STAGE" ]] || exit 2

# Optional --redact-version VERSION is only for an explicit owner-requested removal.
if [[ "${3:-}" == --private-ops && "$#" == 3 ]]; then
  python3 "$STAGE/tools/check_release_notes.py"
  python3 "$STAGE/tools/check_private_release.py" "$PREVIOUS" "$STAGE"
else
  python3 "$STAGE/tools/check_release_notes.py" --previous "$PREVIOUS/gui/content/releases.json" "${@:3}"
fi

if [[ "$MODE" == stage ]]; then
  python3 "$STAGE/tools/build_discovery.py" --check
  # Reuse the private console source from the deployed release, never local visitor data.
  # A deployment may explicitly include updated private dashboard source.
  cp -an "$PREVIOUS/ops_dashboard" "$STAGE/"
  mkdir -p "$STAGE/tools"
  for f in feedback_report.py visitor_report.py; do
    [[ -f "$STAGE/tools/$f" ]] || cp "$PREVIOUS/tools/$f" "$STAGE/tools/$f"
  done
  ln -s "$BASE/shared/feedback" "$STAGE/feedback"
  # The site's favicon is an operational asset excluded from the public source tree.
  if [[ -d "$PREVIOUS/gui/assets" ]]; then cp -a "$PREVIOUS/gui/assets" "$STAGE/gui/"; fi
  python3 -m venv "$STAGE/.venv"
  "$STAGE/.venv/bin/pip" install --quiet -r "$STAGE/requirements-release.txt"
  "$STAGE/.venv/bin/pip" check
  "$STAGE/.venv/bin/pip" freeze > "$STAGE/dependencies.lock.txt"
  chown -R root:root "$STAGE"
  chmod -R a+rX "$STAGE"
  cd "$STAGE"
  AILAB_FEEDBACK_DIR=/tmp/ailab-release-test-feedback MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$STAGE/.venv/bin/python" -m pytest -q
  AILAB_FEEDBACK_DIR=/tmp/ailab-release-test-feedback MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$STAGE/.venv/bin/python" -m pytest -q -m slow tests/test_pages_render.py
  printf '%s\n' "$PREVIOUS" > "$STAGE/previous-release.txt"
  touch "$STAGE/.validated"
  echo "Validated staging release: $STAGE"
elif [[ "$MODE" == activate ]]; then
  [[ -f "$STAGE/.validated" && -f "$STAGE/previous-release.txt" ]] || { echo 'Stage must pass checks first'; exit 2; }
  [[ "$(cat "$STAGE/previous-release.txt")" == "$PREVIOUS" ]] || { echo 'Current release changed; stage again'; exit 2; }
  rollback() {
    echo "Rolling AI Lab back to $PREVIOUS" >&2
    ln -s "$PREVIOUS" "$BASE/current.rollback"
    mv -Tf "$BASE/current.rollback" "$BASE/current"
    systemctl restart ai-lab ai-lab-ops
  }
  if [[ -f "$STAGE/source-manifest.sha256" ]]; then (cd "$STAGE" && sha256sum -c source-manifest.sha256 --quiet); fi
  trap rollback ERR
  ln -s "$STAGE" "$BASE/current.next"
  mv -Tf "$BASE/current.next" "$BASE/current"
  systemctl restart ai-lab ai-lab-ops
  for i in $(seq 1 20); do
    if curl -fsS --max-time 3 http://127.0.0.1:8501/_stcore/health >/dev/null &&
       curl -fsS --max-time 3 http://127.0.0.1:8502/healthz >/dev/null; then break; fi
    sleep 1
  done
  systemctl is-active --quiet ai-lab ai-lab-ops
  curl -fsS --max-time 10 http://127.0.0.1:8501/_stcore/health
  curl -fsS --max-time 10 http://127.0.0.1:8502/healthz
  curl -fsS --max-time 20 https://ai-lab.gokcol.online/_stcore/health
  # The collector cannot approve its own new baseline; activation does so only
  # after manifest verification and health checks, if monitoring is installed.
  if [[ -f /etc/systemd/system/ai-lab-monitor.service ]]; then
    "$STAGE/.venv/bin/python" "$STAGE/tools/collect_ops_security.py" --set-baseline
  fi
  trap - ERR
  echo "Active: $STAGE; rollback: $PREVIOUS"
else
  echo 'Usage: release_update.sh stage|activate RELEASE [--redact-version VERSION | --private-ops]' >&2; exit 2
fi
