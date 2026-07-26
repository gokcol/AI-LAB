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
@pytest.mark.parametrize("button_key", ["_open_terms_en", "_open_terms_tr"])
def test_footer_buttons_open_dialog_without_navigation_error(tmp_path, button_key):
    from tests.test_pages_render import _run

    app = _run(ROOT / "gui" / "views" / "home.py", tmp_path)
    button = next(item for item in app.button if item.key == button_key)
    app = button.click().run()

    assert not app.exception
