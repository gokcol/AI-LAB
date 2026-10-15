"""Protect public Streamlit reruns, including actions on existing WebSockets.

Production uses one Streamlit process. Set AILAB_RESOURCE_LIMITS=1 there; CLI work
and standalone lesson tests remain unrestricted. Counters contain opaque IDs only.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os

import streamlit as st

from core.limits import BusyError, WorkLimiter


_request = ContextVar("ailab_computation", default=None)


@st.cache_resource(show_spinner=False)
def _limiter():
    return WorkLimiter()


def _identity():
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from gui.feedback import client_key

    ctx = get_script_run_ctx(suppress_warning=True)
    return (ctx.session_id, client_key()) if ctx else None


def _retry(error):
    st.info(f"{error} Try again in {error.retry_after} seconds. Your saved learning records are unchanged.")
    st.button("Try again", key="resource_guard_retry")
    st.stop()


@contextmanager
def page_budget():
    identity = _identity() if os.environ.get("AILAB_RESOURCE_LIMITS") == "1" else None
    if identity is None:
        yield
        return
    token = _request.set({"identity": identity, "training": False})
    try:
        with _limiter().render(*identity):
            yield
    except BusyError as error:
        _retry(error)
    finally:
        _request.reset(token)


def limited_training(function):
    """Charge once per rerun, only when a cached calculation actually executes."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        request = _request.get()
        if request is not None and not request["training"]:
            try:
                _limiter().training(*request["identity"])
            except BusyError as error:
                _retry(error)
            request["training"] = True
        return function(*args, **kwargs)
    return wrapped
