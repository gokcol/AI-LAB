"""Small, bounded practice controls shared by the challenge pages."""
import hashlib
import json
import streamlit as st
from gui import progress


def hints(key, steps):
    level_key = key + "_hint_level"
    level = st.session_state.get(level_key, 0)
    if level < len(steps) and st.button("Show next hint", key=key + "_hint"):
        level += 1
        st.session_state[level_key] = level
    for index, hint in enumerate(steps[:level], 1):
        st.info(f"Hint {index}: {hint}")


def prediction_gate(key, *, title, lesson, seed, settings, prompt):
    """Save first, then run on a later interaction. Changing settings starts a new run."""
    serialized = json.dumps(settings, sort_keys=True)
    signature = hashlib.sha256(f"{seed}:{serialized}".encode()).hexdigest()[:16]
    active_key = f"{key}_{signature}"
    run_id = st.session_state.get(active_key)
    record = next((r for r in progress.state()["runs"] if r["id"] == run_id), None)
    if record is None:
        with st.form(active_key + "_prediction"):
            prediction = st.text_area(prompt, max_chars=2000)
            if st.form_submit_button("Record prediction"):
                try:
                    data, run_id = progress.start_run(progress.state(), title=title, lesson=lesson,
                        seed=seed, settings=serialized, prediction=prediction)
                    progress.replace(data)
                    st.session_state[active_key] = run_id
                    record = next(r for r in data["runs"] if r["id"] == run_id)
                    st.success("Prediction saved. Click Run to see the result.")
                except ValueError as exc:
                    st.warning(str(exc))
        # Run requires a separate click, after this render has synced the prediction.
        if record is None:
            return None
    st.caption(f"Recorded prediction: {record['prediction']}")
    if st.button("Run with these settings", key=active_key + "_run"):
        st.session_state[active_key + "_ran"] = True
    return record if st.session_state.get(active_key + "_ran") else None


def save_result(record, result):
    with st.form("result_" + record["id"]):
        explanation = st.text_area("Explain the result: what changed your mind?", max_chars=2000)
        if st.form_submit_button("Save result to notebook"):
            try:
                progress.replace(progress.finish_run(progress.state(), record["id"],
                                                     result=result, explanation=explanation))
                st.success("Saved with your original prediction, seed and settings.")
            except ValueError as exc:
                st.warning(str(exc))
