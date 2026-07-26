"""Feedback form with layered abuse protection (used by the Home page).

Design goals — a public, anonymous-friendly comment box that is hard to abuse:

  * plain text ONLY: no HTML/tags/code/entities/links; control characters rejected;
    whitespace collapsed; message capped at 200 characters (server-side, not just UI)
  * optional name / surname / email, each validated and length-capped
  * bot traps: a CSS-hidden honeypot field and a minimum time-to-fill — both fail
    SILENTLY (the bot sees "success", nothing is stored, nothing is learned)
  * rate limits: per-session cooldown + cap, and a process-wide sliding window with
    a per-client (IP when available, else session) cap — every submit attempt counts,
    so error-probing is throttled too
  * bounded storage: appends JSON-Lines to feedback/feedback.jsonl and refuses when
    the file exceeds a size cap, so disk can never be flooded
  * safe by construction: entries are stored JSON-escaped and are only ever rendered
    back through st.text/st.code (never markdown/HTML), so stored text can't execute

Storage location can be overridden with AILAB_FEEDBACK_DIR (used by tests). On a
self-hosted server (ai-lab.gokcol.online) the feedback file persists on disk; read it
over SSH with  tools/feedback_report.py  (daily/weekly/monthly reports). The in-app
admin view requires AILAB_FEEDBACK_ADMIN=1 — set it ONLY for a local run, never on
the public server, or the inbox becomes public.

Self-hosting notes: app-level limits complement (not replace) a reverse proxy —
put nginx/caddy rate limiting and HTTPS in front, firewall the Streamlit port so
only the proxy can reach it, and set  proxy_set_header X-Forwarded-For
$proxy_add_x_forwarded_for;  so per-client limits see real addresses.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
import threading
import time

import streamlit as st

LAB_ROOT = pathlib.Path(__file__).resolve().parents[1]

# --- policy knobs ----------------------------------------------------------- #
MAX_MSG = 200            # characters, total, after whitespace normalization
MAX_NAME = 50
MAX_EMAIL = 100
MIN_SECONDS_TO_FILL = 3.0    # faster than a human can read + type -> bot
SESSION_COOLDOWN_S = 60      # between submissions from one session
SESSION_MAX = 3              # per session lifetime
WINDOW_S = 3600.0            # sliding window for the global limiter
GLOBAL_MAX_PER_WINDOW = 30   # all clients combined, per window
KEY_MAX_PER_WINDOW = 5       # per client (IP / session), per window
DAILY_MAX = 50               # stored entries per UTC day, all clients combined
MAX_FILE_BYTES = 256_000     # hard cap on the feedback file (~1200 entries)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(
    r"(https?\s*:|www\.|://|ftp\.|\.[a-z]{2,6}/"           # scheme / path forms
    r"|\.(com|net|org|io|co|ru|cn|xyz|info|biz|online|site|top|club|ly|me|tv|cc)\b)",
    re.I)
_ENTITY_RE = re.compile(r"&#?\w{1,12};")
_FORBIDDEN_CHARS = set("<>{}[]`$\\|^~#*_")   # markup/markdown/code machinery


def _dir() -> pathlib.Path:
    override = os.environ.get("AILAB_FEEDBACK_DIR")
    return pathlib.Path(override) if override else LAB_ROOT / "feedback"


def _file() -> pathlib.Path:
    return _dir() / "feedback.jsonl"


# --------------------------------------------------------------------------- #
# Validation — everything is checked server-side; the UI limits are cosmetic
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines) to single spaces and strip."""
    return " ".join(str(text).split())


def validate_message(raw: str) -> tuple[str | None, str | None]:
    """Return (clean_message, None) or (None, error)."""
    msg = normalize(raw)
    if not msg:
        return None, "Please write a message."
    if len(msg) > MAX_MSG:
        return None, f"Message is {len(msg)} characters — the limit is {MAX_MSG}."
    if any(ch in _FORBIDDEN_CHARS or ord(ch) < 32 for ch in msg):
        return None, "Plain text only, please — no code, tags, or markup characters."
    if _URL_RE.search(msg):
        return None, "Links are not allowed in feedback."
    if _ENTITY_RE.search(msg):
        return None, "Plain text only, please — no HTML entities."
    return msg, None


def validate_name(raw: str, label: str) -> tuple[str | None, str | None]:
    """Optional field: empty is fine. Letters (any script), spaces, ' - . only."""
    name = normalize(raw)
    if not name:
        return "", None
    if len(name) > MAX_NAME:
        return None, f"{label} is too long (max {MAX_NAME} characters)."
    if not all(ch.isalpha() or ch in " '-." for ch in name):
        return None, f"{label} may only contain letters, spaces, and ' - ."
    return name, None


def validate_email(raw: str) -> tuple[str | None, str | None]:
    email = normalize(raw)
    if not email:
        return "", None
    if len(email) > MAX_EMAIL or not _EMAIL_RE.fullmatch(email):
        return None, "That email address doesn't look valid."
    return email, None


# --------------------------------------------------------------------------- #
# Rate limiting — a process-wide sliding window (Streamlit runs one process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _limiter():
    return {"lock": threading.Lock(), "events": collections.deque()}


def _xff_client(headers) -> str | None:
    """Take the LAST X-Forwarded-For entry: with nginx's
    $proxy_add_x_forwarded_for the last hop is appended by *your* proxy and is the
    address that actually connected; earlier entries are client-supplied junk."""
    try:
        xff = headers.get("X-Forwarded-For") if headers else None
    except Exception:
        return None
    if not xff:
        return None
    parts = [p.strip() for p in str(xff).split(",") if p.strip()]
    return parts[-1] if parts else None


def client_key() -> str:
    """Best client identifier available, hashed (we never store a raw IP):
    X-Forwarded-For (self-hosted behind nginx) > socket IP > session id."""
    raw = None
    try:
        raw = _xff_client(st.context.headers)
    except Exception:
        raw = None
    if not raw:
        try:
            raw = st.context.ip_address       # None on localhost / bare mode
        except Exception:
            raw = None
    if not raw:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            raw = ctx.session_id if ctx else "unknown"
        except Exception:
            raw = "unknown"
    return hashlib.sha256(str(raw).encode()).hexdigest()[:16]


def check_and_record(key: str, now: float | None = None) -> tuple[bool, str | None]:
    """Sliding-window limiter. Records EVERY submit attempt (valid or not), so
    probing with garbage costs the attacker their budget too."""
    now = time.time() if now is None else now
    lim = _limiter()
    with lim["lock"]:
        events = lim["events"]
        while events and events[0][0] < now - WINDOW_S:
            events.popleft()
        if len(events) >= GLOBAL_MAX_PER_WINDOW:
            return False, "The feedback box is busy right now — please try again later."
        if sum(1 for _, k in events if k == key) >= KEY_MAX_PER_WINDOW:
            return False, "You've reached the feedback limit for now — thank you! Try again later."
        events.append((now, key))
        return True, None


# --------------------------------------------------------------------------- #
# Storage — bounded, escaped, append-only
# --------------------------------------------------------------------------- #
_write_lock = threading.Lock()
_daily = {"day": None, "count": 0}          # per-process cache of today's stored count


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _count_today_in_file(f: pathlib.Path, today: str) -> int:
    if not f.exists():
        return 0
    n = 0
    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line[2:12] == today or today in line[:30]:
                    n += 1
    except OSError:
        return 0
    return n


def _append(record: dict) -> tuple[bool, str | None]:
    """Append one entry, enforcing the size cap and the DAILY_MAX cap atomically.
    The daily counter is cached per process and re-seeded from the file on the
    first write of each UTC day, so it survives restarts (persistent disk)."""
    try:
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        f = _file()
        with _write_lock:
            today = _today()
            if _daily["day"] != today:
                _daily["day"] = today
                _daily["count"] = _count_today_in_file(f, today)
            if _daily["count"] >= DAILY_MAX:
                return False, ("Today's feedback box is full (daily limit reached) — "
                               "please come back tomorrow, or open a GitHub issue.")
            if f.exists() and f.stat().st_size > MAX_FILE_BYTES:
                return False, "The feedback box is full — thank you for trying! Please use GitHub instead."
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            _daily["count"] += 1
        return True, None
    except OSError:
        return False, "Couldn't save feedback right now — please try again later."


# --------------------------------------------------------------------------- #
# The submit pipeline
# --------------------------------------------------------------------------- #
_THANKS = "Thank you — your feedback was received! 🧠"


def submit(name: str, surname: str, email: str, message: str,
           honeypot: str, first_seen: float) -> tuple[str, str]:
    """Run the full protection pipeline. Returns (level, text) where level is one of
    'success' | 'warning' | 'error'. Bot detections return 'success' with nothing
    stored, so automated senders learn nothing."""
    now = time.time()

    # 1 · bot traps — silent discard
    if normalize(honeypot):
        return "success", _THANKS
    if first_seen and (now - first_seen) < MIN_SECONDS_TO_FILL:
        return "success", _THANKS

    # 2 · session limits
    ss = st.session_state
    if ss.get("fb_count", 0) >= SESSION_MAX:
        return "warning", "You've sent the maximum feedback for this session — thank you!"
    since = now - ss.get("fb_last", 0.0)
    if since < SESSION_COOLDOWN_S:
        return "warning", f"Please wait {int(SESSION_COOLDOWN_S - since) + 1}s before sending again."

    # 3 · global limiter (counts this attempt whatever happens next)
    allowed, why = check_and_record(client_key(), now)
    if not allowed:
        return "warning", why

    # 4 · validation
    msg, err = validate_message(message)
    if err:
        return "error", err
    nm, err = validate_name(name, "Name")
    if err:
        return "error", err
    sn, err = validate_name(surname, "Surname")
    if err:
        return "error", err
    em, err = validate_email(email)
    if err:
        return "error", err

    # 5 · store (escaped; client key hashed; no raw IP anywhere)
    ok, err = _append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
        "name": nm, "surname": sn, "email": em,
        "message": msg,
        "client": client_key(),
    })
    if not ok:
        return "warning", err

    ss["fb_count"] = ss.get("fb_count", 0) + 1
    ss["fb_last"] = now
    return "success", _THANKS


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_form() -> None:
    st.session_state.setdefault("fb_first_seen", time.time())

    with st.form("fb_form", clear_on_submit=True, border=True):
        c = st.columns(3)
        name = c[0].text_input("Name *(optional)*", max_chars=MAX_NAME, key="fb_name")
        surname = c[1].text_input("Surname *(optional)*", max_chars=MAX_NAME, key="fb_surname")
        email = c[2].text_input("Email *(optional)*", max_chars=MAX_EMAIL, key="fb_email",
                                help="Only if you'd like a reply. Never shared.")
        message = st.text_area(
            f"Message · plain text · max {MAX_MSG} characters", max_chars=MAX_MSG,
            height=90, key="fb_message",
            placeholder="What did you like? What's confusing? What should the lab teach next?",
        )
        # Honeypot — hidden by CSS (.st-key-fb_hp in ui.inject_theme). Humans never
        # see it; anything typed here marks the submission as automated.
        honeypot = st.text_input("Leave this field empty", key="fb_hp",
                                 label_visibility="collapsed")
        submitted = st.form_submit_button("Send feedback", icon=":material/send:",
                                          type="primary")

    st.caption("No links, code, or HTML — plain text only. Optional fields are used solely "
               "to read and reply to feedback, are never shared, and you can ask for removal "
               "any time via GitHub. Rate limits apply.")

    if submitted:
        level, text = submit(name, surname, email, message,
                             honeypot, st.session_state.get("fb_first_seen", 0.0))
        {"success": st.success, "warning": st.warning, "error": st.error}[level](text)

    # Local-only admin view (env var is never set on the public deploy).
    if os.environ.get("AILAB_FEEDBACK_ADMIN") == "1":
        with st.expander("🗂 Feedback inbox (admin — local only)"):
            f = _file()
            if not f.exists():
                st.caption("No feedback yet.")
            else:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
                st.caption(f"{len(lines)} entr{'y' if len(lines) == 1 else 'ies'} · "
                           f"{f.stat().st_size:,} bytes")
                for line in lines[-50:][::-1]:
                    try:
                        r = json.loads(line)
                        who = " ".join(x for x in (r.get("name"), r.get("surname")) if x) or "anonymous"
                        st.text(f"[{r.get('ts','?')}] {who}"
                                + (f" <{r['email']}>" if r.get("email") else "")
                                + f"\n  {r.get('message','')}")
                    except json.JSONDecodeError:
                        st.text(f"(unparseable line) {line[:120]}")
                st.download_button("Download feedback.jsonl", f.read_bytes(),
                                   file_name="feedback.jsonl", mime="application/jsonl")
