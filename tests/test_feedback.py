"""Protection-layer tests for the public feedback form (gui/feedback.py).

Pure-Python unit tests (no Streamlit runtime needed) covering validation, the
sliding-window limiter, the daily cap, size cap, and the XFF client identifier.
"""

import importlib
import sys
import time
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "gui"
sys.path.insert(0, str(GUI))


@pytest.fixture()
def fb(tmp_path, monkeypatch):
    monkeypatch.setenv("AILAB_FEEDBACK_DIR", str(tmp_path))
    mod = importlib.import_module("feedback")
    importlib.reload(mod)          # pick up the patched dir / fresh module state
    return mod


# --- message validation ---------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "", "   ", "x" * 201,
    "nice <b>bold</b>", "<script>alert(1)</script>", "run `code`",
    "**markdown**", "a $x$ b", "visit http://spam.example", "go to www.spam.io",
    "buy at deals.online now", "&lt;entity&gt;", "hi\x07bell",
])
def test_message_rejected(fb, bad):
    clean, err = fb.validate_message(bad)
    assert clean is None and err


@pytest.mark.parametrize("good", [
    "Great lab, the XOR demo finally made backprop click!",
    "x" * 200,
    "Çok güzel olmuş, teşekkürler — öğretici ve net.",
    "Please cover diffusion models next. 10/10 would learn again",
])
def test_message_accepted(fb, good):
    clean, err = fb.validate_message(good)
    assert err is None and clean


def test_whitespace_collapsed(fb):
    assert fb.validate_message("a\n\n b\t c")[0] == "a b c"


# --- name / email ---------------------------------------------------------- #
def test_names(fb):
    assert fb.validate_name("", "Name") == ("", None)
    assert fb.validate_name("O'Brien-Smith", "Name")[1] is None
    assert fb.validate_name("Gökçöl", "Name")[1] is None
    assert fb.validate_name("Bot123", "Name")[1] is not None
    assert fb.validate_name("a" * 51, "Name")[1] is not None


def test_email(fb):
    assert fb.validate_email("")[1] is None
    assert fb.validate_email("a.b+c@ex-ample.co")[1] is None
    assert fb.validate_email("nope")[1] is not None


# --- limiter --------------------------------------------------------------- #
def test_sliding_window(fb):
    lim = fb._limiter(); lim["events"].clear()
    t0 = 1_000_000.0
    allowed = sum(fb.check_and_record("A", t0 + i)[0] for i in range(10))
    assert allowed == fb.KEY_MAX_PER_WINDOW
    total = allowed + sum(fb.check_and_record(f"c{i}", t0 + 50 + i)[0] for i in range(40))
    assert total == fb.GLOBAL_MAX_PER_WINDOW
    assert fb.check_and_record("A", t0 + fb.WINDOW_S + 100)[0]  # window slid


def test_xff_last_entry_wins(fb):
    # $proxy_add_x_forwarded_for appends the real client last; earlier are spoofable
    assert fb._xff_client({"X-Forwarded-For": "6.6.6.6, 1.2.3.4"}) == "1.2.3.4"
    assert fb._xff_client({}) is None


# --- storage --------------------------------------------------------------- #
def test_daily_cap_survives_restart(fb):
    day = time.strftime("%Y-%m-%d", time.gmtime())
    f = fb._file(); f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fh:
        for i in range(fb.DAILY_MAX):
            fh.write(f'{{"ts": "{day} 01:00:00 UTC", "message": "m{i}"}}\n')
    fb._daily.update(day=None, count=0)        # simulate a process restart
    ok, err = fb._append({"ts": f"{day} 02:00:00 UTC", "message": "over"})
    assert not ok and "daily" in err.lower()


def test_size_cap(fb):
    f = fb._file(); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x" * (fb.MAX_FILE_BYTES + 1))
    fb._daily.update(day=None, count=0)
    ok, err = fb._append({"ts": "2026-07-26 00:00:00 UTC", "message": "hi"})
    assert not ok and "full" in err.lower()


def test_consent_required_only_when_personal_data_given(fb, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    old = time.time() - 30

    def send(**kw):
        st.session_state.clear()
        fb._limiter()["events"].clear()   # this probe sends more than the hourly quota
        return fb.submit(honeypot="", first_seen=old, message="A genuinely useful note.",
                         **{"name": "", "surname": "", "email": "", "consent": False, **kw})

    # anonymous: nothing to consent to, so no tick is demanded
    assert send()[0] == "success"
    # any personal field without consent is refused, and refused as an *error* so the
    # form keeps what was typed
    for field in ("name", "surname", "email"):
        val = "orhan@example.com" if field == "email" else "Orhan"
        level, text = send(**{field: val})
        assert level == "error" and "consent" in text.lower(), field
        # ...and with the tick it goes through
        assert send(consent=True, **{field: val})[0] == "success", field


def test_bot_traps_are_silent(fb, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    # honeypot filled -> reported success, nothing stored
    level, _ = fb.submit("", "", "", "hi", honeypot="x", first_seen=time.time() - 30)
    assert level == "success"
    assert not fb._file().exists() or fb._file().read_text().strip() == ""
    # too fast -> same
    level, _ = fb.submit("", "", "", "hi", honeypot="", first_seen=time.time())
    assert level == "success"
