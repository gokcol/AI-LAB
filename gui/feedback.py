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

import base64
import collections
import hashlib
import io
import random
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
IP_DAILY_MAX = 5             # stored entries per client per UTC day (then blocked)
DAILY_MAX = 50               # stored entries per UTC day, all clients combined
MAX_FILE_BYTES = 256_000     # hard cap on the feedback file (~1200 entries)

# Data-protection facts stated on the form itself, so the notice and the code cannot
# drift apart. Personal data is optional; consent is only required when it is given.
RETENTION_DAYS = 365         # maximum retention for an entry carrying a name/email
# Where the data physically sits. This appears in a privacy notice, so it must be true:
# set AILAB_DATA_REGION on the server to the actual Linode region (e.g. "Frankfurt,
# Germany"). The default only claims what is verifiable from the IP allocation — the
# address block is registered in the RIPE (European) region.
DATA_LOCATION = os.environ.get("AILAB_DATA_REGION", "the European Union")

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
# Graphical CAPTCHA — rendered locally with matplotlib as a PNG.
# No third-party service, no cookies, no tracking: the challenge is drawn on this
# server and the answer lives only in the visitor's session. Characters are drawn
# into a raster image (not as SVG/DOM text), so a scraper cannot simply read them
# out of the page source.
# --------------------------------------------------------------------------- #
CAPTCHA_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no confusable I/O/0/1
CAPTCHA_LEN = 5


def _draw_captcha(text: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(text)                     # deterministic per challenge
    fig, ax = plt.subplots(figsize=(2.9, 0.95), dpi=110)
    ax.set_xlim(0, len(text) + 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F4F7FB")
    for i, ch in enumerate(text):                 # jittered, rotated glyphs
        ax.text(i + 0.45 + rng.uniform(-0.09, 0.09), 0.42 + rng.uniform(-0.12, 0.12), ch,
                fontsize=rng.randint(23, 29),
                rotation=rng.uniform(-26, 26),
                color=rng.choice(["#17324F", "#1D4ED8", "#8A2351", "#0E5E45"]),
                fontweight="bold", ha="center", va="center",
                family=rng.choice(["DejaVu Sans", "DejaVu Serif"]))
    for _ in range(4):                            # noise strokes
        xs = [rng.uniform(0, len(text) + 0.4) for _ in range(3)]
        ys = [rng.uniform(0, 1) for _ in range(3)]
        ax.plot(xs, ys, lw=rng.uniform(0.8, 1.6), alpha=0.45,
                color=rng.choice(["#9CA3AF", "#93B4E8", "#C0507A"]))
    for _ in range(160):                          # speckle
        ax.plot(rng.uniform(0, len(text) + 0.4), rng.uniform(0, 1), ".",
                ms=rng.uniform(0.6, 1.8), color="#9CA3AF", alpha=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return buf.getvalue()


def new_captcha() -> None:
    """Create a fresh challenge and remember only its hash in the session."""
    text = "".join(random.choice(CAPTCHA_CHARS) for _ in range(CAPTCHA_LEN))
    st.session_state["fb_captcha_png"] = _draw_captcha(text)
    st.session_state["fb_captcha_hash"] = hashlib.sha256(text.encode()).hexdigest()


def check_captcha(answer: str) -> bool:
    want = st.session_state.get("fb_captcha_hash")
    if not want:
        return False
    got = hashlib.sha256(normalize(answer).upper().encode()).hexdigest()
    return hmac_compare(got, want)


def hmac_compare(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a, b)


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


def _ip_day_count(f: pathlib.Path, today: str, client: str) -> int:
    """How many entries this client already stored today (read from disk, so the cap
    survives restarts and cannot be reset by dropping the session cookie)."""
    if not client or not f.exists():
        return 0
    n = 0
    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if today in line and client in line:
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
            if _ip_day_count(f, today, record.get("client", "")) >= IP_DAILY_MAX:
                return False, ("You have reached today's feedback limit from this connection "
                               "— thank you! Please come back tomorrow, or open a GitHub issue.")
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
           honeypot: str, first_seen: float, captcha: str = "",
           consent: bool = False) -> tuple[str, str]:
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

    # 4 · CAPTCHA — checked before validation so a bot cannot use the error messages
    #     to probe the validator. Only enforced when a challenge was actually issued.
    if st.session_state.get("fb_captcha_hash") and not check_captcha(captcha):
        new_captcha()
        return "error", "The characters did not match — please try the new image."

    # 5 · validation
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

    # 6 · consent — required only when personal data was actually entered. An anonymous
    #     note carries nothing to consent to, so it is never blocked by a tick box.
    if (nm or sn or em) and not consent:
        return "error", ("You entered a name or email — please tick the consent box under "
                         "those fields so they may be stored, or clear them to send the "
                         "message anonymously. Your text is kept either way.")

    # 7 · store (escaped; client key hashed; no raw IP anywhere)
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
_FIELD_KEYS = ("fb_message", "fb_name", "fb_surname", "fb_email", "fb_consent", "fb_hp")


def render_form() -> None:
    # A rejected submission must never cost the visitor their typing. Streamlit can only
    # clear a widget BEFORE it is instantiated, so the run that handled the submit left a
    # flag here instead of wiping the form itself:
    #   "all"     – it was really stored, start a clean form
    #   "captcha" – rejected; keep every word, but the challenge image has rotated so the
    #               answer typed against the old one is stale and must not linger.
    reset = st.session_state.pop("fb_reset", None)
    if reset:
        st.session_state.pop("fb_captcha", None)
    if reset == "all":
        for k in _FIELD_KEYS:
            st.session_state.pop(k, None)

    st.session_state.setdefault("fb_first_seen", time.time())
    if "fb_captcha_png" not in st.session_state:
        new_captcha()

    st.markdown(
        '<div class="ailab-fb-head">'
        "<h4>💬 Tell me what you think</h4>"
        "<p>Spotted a mistake? Something explained badly? A topic you wish were here? "
        "These notes get better because people say so — one line is plenty.</p>"
        f'<span class="ailab-chip">✍️ plain text</span>'
        f'<span class="ailab-chip">{MAX_MSG} characters</span>'
        '<span class="ailab-chip">name &amp; email optional</span>'
        '<span class="ailab-chip">🔒 never shared</span>'
        '<span class="ailab-chip">🤖 bot-checked</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # clear_on_submit stays False on purpose: it clears on EVERY submit, including a
    # rejected one, so a mistyped email used to cost the visitor their whole message.
    # Clearing is now conditional on the entry actually being stored (see below).
    with st.form("fb_form", clear_on_submit=False, border=False):
        message = st.text_area(
            "Your message",
            max_chars=MAX_MSG, height=120, key="fb_message",
            placeholder="e.g. “The XOR walkthrough finally made backprop click — but the "
                        "softmax section lost me.”",
            help=f"Plain text only — no links, code or HTML. Up to {MAX_MSG} characters.",
        )
        with st.expander("Add your name or email *(optional — only if you'd like a reply)*"):
            c = st.columns(3)
            name = c[0].text_input("Name", max_chars=MAX_NAME, key="fb_name",
                                   placeholder="optional")
            surname = c[1].text_input("Surname", max_chars=MAX_NAME, key="fb_surname",
                                      placeholder="optional")
            email = c[2].text_input("Email", max_chars=MAX_EMAIL, key="fb_email",
                                    placeholder="optional")
            # Informed consent, and only when there is something to consent to: leave
            # these three boxes empty and no personal data is processed at all, so no
            # tick is required. GDPR Art. 4(11)/7 and KVKK Art. 5 both want a specific,
            # informed, unambiguous, affirmative act — never a pre-ticked box.
            st.caption(
                f"**Why this is asked.** If you fill any box above, that name and address "
                f"are stored on this site's own server (in {DATA_LOCATION}), used **only** "
                "to reply to you, never shown on the site, never shared, sold or used for "
                "any mailing, and deleted on request or after "
                f"{RETENTION_DAYS} days at the latest. Leave them empty and nothing "
                "identifying you is kept — the note is stored anonymously."
            )
            consent = st.checkbox(
                "I agree that the name and email I entered above may be stored so the "
                "author can reply to me.",
                key="fb_consent",
            )
        # Honeypot — hidden by CSS (.st-key-fb_hp). Humans never see it; anything typed
        # here marks the submission as automated.
        honeypot = st.text_input("Leave this field empty", key="fb_hp",
                                 label_visibility="collapsed")

        cap_l, cap_r = st.columns([0.44, 0.56])
        with cap_l:
            st.markdown("<div style='font-size:.83rem;color:#56697F;margin-bottom:.25rem'>"
                        "Type the characters you see:</div>", unsafe_allow_html=True)
            st.image(st.session_state["fb_captcha_png"])
        with cap_r:
            captcha = st.text_input("Characters", key="fb_captcha",
                                    max_chars=CAPTCHA_LEN + 3,
                                    placeholder=f"{CAPTCHA_LEN} characters",
                                    help="Not case-sensitive. A new image appears after "
                                         "each attempt.")
        # NOTE: keep this at the form's top level. Nesting a form_submit_button inside
        # st.columns makes the submission untestable via AppTest, and an unverifiable
        # submit path is not worth a nicer row.
        st.markdown("<div style='font-size:.82rem;color:#7A8A99;margin:-.2rem 0 .5rem'>"
                    "Anonymous is completely fine — just write the message.</div>",
                    unsafe_allow_html=True)
        submitted = st.form_submit_button("Send feedback", icon=":material/send:",
                                          type="primary")

    if submitted:
        before = st.session_state.get("fb_count", 0)
        level, text = submit(name, surname, email, message, honeypot,
                             st.session_state.get("fb_first_seen", 0.0), captcha, consent)
        # Celebrate only when something was really stored: the bot traps also report
        # "success" (deliberately), and they must not look different to a machine.
        stored = level == "success" and st.session_state.get("fb_count", 0) > before
        if level != "error":          # errors already refreshed the challenge
            new_captcha()
        if stored:
            # Restart the "too fast to be human" clock for the next message. Deliberately
            # NOT done on a rejection: someone correcting a typo in two seconds would then
            # trip the bot trap and have their fixed message silently dropped.
            st.session_state["fb_first_seen"] = time.time()
        st.session_state["fb_notice"] = (level, text, stored)
        st.session_state["fb_reset"] = "all" if stored else "captcha"
        st.rerun()

    notice = st.session_state.pop("fb_notice", None)
    if notice:
        level, text, celebrate = notice
        {"success": st.success, "warning": st.warning, "error": st.error}[level](text)
        if celebrate:
            st.balloons()

    st.caption("If the image is hard to read for any reason, please open a GitHub issue "
               "instead — the link is in the About tab. ")
    st.caption("Your note is stored on the site's own server and read only by the author. "
               "No links, code or HTML — plain text keeps the box safe for everyone. "
               "Ask any time (via GitHub) and it will be deleted. Rate limits apply.")

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
