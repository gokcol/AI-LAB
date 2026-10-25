"""Regression checks for the footer Terms dialog routing."""

from __future__ import annotations

import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_footer_does_not_navigate_through_streamlit_router():
    source = (ROOT / "gui" / "views" / "home.py").read_text(encoding="utf-8")

    assert "[📜 Terms of use & data](?terms=" not in source
    assert "[🇹🇷 Kullanım Koşulları ve Veri Kullanımı](?terms=" not in source
    assert 'on_click=_request_terms' in source
    assert 'st.session_state.pop("_terms_request", None)' in source


@pytest.mark.slow
@pytest.mark.parametrize("requested_language", ["en", "tr"])
def test_footer_buttons_open_dialog_without_navigation_error(tmp_path, monkeypatch, requested_language):
    from streamlit.testing.v1 import AppTest
    from gui import i18n, progress

    monkeypatch.syspath_prepend(str(ROOT / "gui"))
    monkeypatch.setenv("AILAB_FEEDBACK_DIR", str(tmp_path))
    monkeypatch.setattr(progress, "_component", None)
    # Keep real navigation available on both the initial render and button rerun.
    app = AppTest.from_file(str(ROOT / "gui" / "app.py"), default_timeout=60).run()
    assert not app.exception
    app.selectbox(key="app_language_dropdown").select(requested_language).run()
    assert not app.exception

    # Turkish selection currently keeps English content and explains the fallback.
    assert "_open_terms_tr" not in {button.key for button in app.button}
    if requested_language == "tr":
        assert any(i18n.TURKISH_UNAVAILABLE_NOTICE in warning.value for warning in app.warning)

    app.button(key="_open_terms_en").click().run()

    assert not app.exception
    dialogs = app.get("dialog")
    assert len(dialogs) == 1
    assert dialogs[0].proto.dialog.is_open
    assert dialogs[0].proto.dialog.title == "📜 Terms of use & privacy"
    assert app.session_state["reader_lang"] == "en"
