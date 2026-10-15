"""Learning checkpoints and a portable, browser-saved experiment notebook."""
import copy
import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from gui import curriculum, progress
from gui.guided_content import LESSONS, practice_summary

st.title("My learning & experiment notebook")
st.caption("Your checkpoints stay in this browser. They are processed in your current app session, not stored as learner records on the server. Export a backup to move devices. Avoid sensitive information.")
data = progress.state()
last = data.get("last_topic")
if last in curriculum.CATALOG:
    st.page_link(curriculum.CATALOG[last].path, label=f"Continue: {curriculum.CATALOG[last].title}", icon=":material/play_arrow:")
else:
    st.page_link("views/start_here.py", label="Begin your first 20-minute session", icon=":material/flag:")
checkpoints, notebook, backup = st.tabs(["Checkpoints", "Experiment notebook", "Export / import"])
with checkpoints:
    track = st.selectbox("Track", ["All", "ANN", "ML", "Math"])
    bookmarked = st.checkbox("Bookmarks only")
    for topic in curriculum.TOPICS:
        record = data["topics"].get(topic.key, {})
        if track != "All" and topic.track != track or bookmarked and not record.get("bookmark"):
            continue
        cols = st.columns([3, 2])
        cols[0].page_link(topic.path, label=("★ " if record.get("bookmark") else "") + topic.title)
        cols[1].caption(record.get("stage", "Not started"))
        if record.get("note"):
            with st.expander(f"My note · {topic.title}"):
                st.write(record["note"])
    st.subheader("Guided exercise results")
    for key, lesson in LESSONS.items():
        if track != "All" and lesson["track"] != track:
            continue
        counts = practice_summary(key, data.get("practice", {}))
        st.write(f"**{lesson['title']}:** {counts['solved']}/8 solved · {counts['independent']}/8 without hints or revealed answers")
    st.caption("These results count exercise types across practice sets, separately from self-assessed checkpoints. They are practice records, not a supervised assessment.")
    st.caption("Explored = tried it; Explained = can explain without notes; Rebuilt = reconstructed independently. These are self-assessments, not grades.")
with notebook:
    st.write("Record a prediction before running an experiment. Keep the seed and settings so you can reproduce it.")
    with st.form("notebook_entry", clear_on_submit=True):
        title = st.text_input("Experiment title", max_chars=150)
        lesson = st.selectbox("Related lesson", list(curriculum.CATALOG), format_func=lambda k: curriculum.CATALOG[k].title)
        c1,c2 = st.columns(2)
        seed = c1.text_input("Seed", max_chars=30)
        settings = c2.text_input("Settings (learning rate, model, data split…)", max_chars=1000)
        prediction = st.text_area("Prediction", max_chars=2000)
        if st.form_submit_button("Save prediction before running"):
            try:
                updated, _ = progress.start_run(data, title=title, lesson=lesson, seed=seed,
                                                settings=settings, prediction=prediction)
                progress.replace(updated)
                data = progress.state()
                st.success("Prediction recorded. Run your experiment, then add the result below.")
            except ValueError as exc:
                st.warning(str(exc))
    pending = [r for r in data["runs"] if r.get("prediction_saved_at") and not r.get("completed_at")]
    if pending:
        chosen = st.selectbox("Finish a recorded experiment", range(len(pending)),
                              format_func=lambda i: pending[i]["title"] + " · " + pending[i]["created"][:19])
        original = pending[chosen]
        st.write(f"**Recorded prediction:** {original['prediction']}")
        st.caption(f"Seed: {original['seed']} · Settings: {original['settings']}")
        with st.form("notebook_result_" + original["id"], clear_on_submit=True):
            result = st.text_area("Observed result", max_chars=2000)
            explanation = st.text_area("Explanation / what changed my mind", max_chars=2000)
            if st.form_submit_button("Save result and explanation"):
                try:
                    progress.replace(progress.finish_run(data, original["id"], result=result, explanation=explanation))
                    data = progress.state()
                    st.success("Result saved; your original prediction is preserved.")
                except ValueError as exc:
                    st.warning(str(exc))
    runs = data["runs"]
    if runs:
        selected = st.multiselect("Compare up to two runs", range(len(runs)), max_selections=2,
                                  format_func=lambda i: f"{i+1} · {runs[i]['title']}")
        if selected:
            st.dataframe(pd.DataFrame([runs[i] for i in selected]).drop(columns=["id"]).T, width="stretch")
        for run in reversed(runs[-20:]):
            with st.expander(f"{run['title']} · {run['created'][:10]}"):
                for label in ("seed", "settings", "prediction", "prediction_saved_at", "result", "explanation", "completed_at"):
                    st.write(f"**{label.title()}**: {run.get(label, '')}")
with backup:
    st.download_button("Export learning file", json.dumps(data, ensure_ascii=False, indent=2),
                       file_name="ai-lab-learning.json", mime="application/json")
    uploaded = st.file_uploader("Restore an AI Lab learning file (max 500 KB)", type=["json"], max_upload_size=1)
    if uploaded is not None:
        try:
            restored = progress.validate(uploaded.getvalue())
            st.caption(f"{len(restored['topics'])} checkpoints · {len(restored['runs'])} notebook entries")
            if st.button("Merge imported learning"):
                merged = copy.deepcopy(data)
                for entry in restored["topics"].values():
                    entry["updated"] = datetime.now(timezone.utc).isoformat()
                merged["topics"].update(restored["topics"])
                merged["practice"] = progress.merge_practice(merged.get("practice", {}), restored.get("practice", {}))
                indexed = {r["id"]: r for r in merged["runs"]}
                indexed.update({r["id"]: r for r in restored["runs"]})
                merged["runs"] = list(indexed.values())
                merged["last_topic"] = restored["last_topic"] or merged["last_topic"]
                progress.replace(merged)
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))
