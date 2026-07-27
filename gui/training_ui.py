"""Small, shared UI pieces for the step-by-step training lessons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import i18n


PHASES = ("1 · Forward", "2 · Loss", "3 · Backward", "4 · Update", "5 · Evaluate")


def action_bar(key: str):
    """Return the requested replay action without hiding a training loop in the UI."""
    cols = st.columns([1.05, 1.05, 1.0, 1.0, .85])
    if cols[0].button("One training step", key=f"{key}_one", icon=":material/skip_next:"):
        return "one"
    if cols[1].button("One epoch", key=f"{key}_epoch", icon=":material/refresh:"):
        return "epoch"
    if cols[2].button("Train 10", key=f"{key}_ten", icon=":material/fast_forward:"):
        return "ten"
    if cols[3].button("Train to completion", key=f"{key}_all", icon=":material/play_arrow:"):
        return "all"
    if cols[4].button("Reset", key=f"{key}_reset", icon=":material/restart_alt:"):
        return "reset"
    return None


def phase_picker(key: str):
    # Keep the returned value canonical English even when i18n localizes visible labels.
    # Otherwise translated widget options would no longer match the phase comparisons in views.
    selected = st.segmented_control(
        "Follow this checkpoint", list(range(len(PHASES))), default=len(PHASES) - 1,
        format_func=lambda i: i18n.localize(PHASES[i]), key=f"{key}_phase",
    )
    return PHASES[len(PHASES) - 1 if selected is None else int(selected)]


def checkpoint_picker(key: str, history):
    if not history:
        return None
    if len(history) == 1:
        st.caption(f"Checkpoint: step 1 · loss {history[0].loss:.6f}")
        return history[0]
    if len(history) > 80:
        index = st.slider("Checkpoint", 0, len(history) - 1, len(history) - 1,
                          key=f"{key}_checkpoint_index")
        st.caption(f"step {index + 1} · loss {history[index].loss:.6f}")
        return history[index]
    labels = [f"step {i + 1} · loss {s.loss:.4f}" for i, s in enumerate(history)]
    choice = st.select_slider("Checkpoint", labels, value=labels[-1], key=f"{key}_checkpoint")
    return history[labels.index(choice)]


def parameter_ledger(names, step, precision=5):
    rows = []
    for name, before, grad, delta, after in zip(names, np.ravel(step.parameters_before),
                                                 np.ravel(step.gradients), np.ravel(step.delta),
                                                 np.ravel(step.parameters_after)):
        rows.append({"parameter": name, "before": before, "gradient": grad,
                     "change": delta, "after": after})
    st.markdown("**Parameter ledger** — `after = before − learning rate × gradient`")
    st.dataframe(pd.DataFrame(rows).round(precision), hide_index=True, width="stretch")


def verification_panel(loss, analytic_grad, numerical_grad=None, extra=()):
    checks = [f"Loss reconstructed from this checkpoint: **{float(loss):.6f}**"]
    if numerical_grad is not None:
        err = float(np.max(np.abs(np.ravel(analytic_grad) - np.ravel(numerical_grad))))
        checks.append(f"Largest finite-difference gradient error: **{err:.2e}**")
    checks.extend(extra)
    with st.expander("✓ Verify this checkpoint"):
        for check in checks:
            st.markdown(f"- {check}")


def check_number(key: str, prompt: str, answer: float, tolerance=1e-3):
    """An optional learner calculation check; no answer is persisted outside session state."""
    guess = st.number_input(prompt, value=None, step=.01, key=f"{key}_guess")
    if guess is not None:
        if abs(float(guess) - float(answer)) <= tolerance:
            st.success("Correct within the displayed rounding tolerance.")
        else:
            st.info(f"Not quite. Full-precision answer: `{answer:.6f}`.")
