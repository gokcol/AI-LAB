"""Shared learner orientation and checkpoint controls."""
from urllib.parse import urlencode
import streamlit as st
from gui import curriculum, progress, i18n


def orientation(key):
    topic = curriculum.CATALOG.get(key)
    if topic is None:
        return
    st.caption(f"{topic.track} · about {topic.minutes} minutes · Goal: {topic.objective}")
    with st.expander("Before you begin · prerequisites and scope"):
        for pre in topic.prerequisites:
            item = curriculum.CATALOG[pre]
            st.page_link(item.path, label=item.title)
        if not topic.prerequisites:
            st.write("No earlier lab lesson required. Work through the example one number at a time.")
        st.write(curriculum.ASSUMPTIONS.get(key, topic.assumption))
        st.caption(f"Guide updated: {topic.updated} · Human reviewed: {topic.reviewed or 'not yet reviewed'}")
        if i18n.MULTILINGUAL:
            st.caption("Turkish: " + topic.turkish_status)
        st.page_link("views/beginner_path.py", label=(
            "Complete beginner path in English and Turkish (review pending)" if i18n.MULTILINGUAL
            else "Complete beginner path (review pending)"))
    bridges = curriculum.EXPERIMENT_BRIDGES.get(key, ())
    if bridges:
        with st.expander("Rebuild this concept in a runnable experiment"):
            for experiment in bridges:
                st.link_button(f"{experiment} · instructions, code and run", f"/experiments?experiment={experiment}")
            st.caption("e12/e13 and e21 train locally; smaller exercises run here. Make a prediction and record your result in My learning.")



def checkpoint(key):
    topic = curriculum.CATALOG.get(key)
    if topic is None:
        return
    st.divider()
    with st.expander("My checkpoint · bookmark, explain, revisit"):
        st.caption("Saved in this browser and processed in your current app session. No account or server-side learning history. Export a backup in My learning.")
        record = progress.state()["topics"].get(key, {})
        with st.form(f"checkpoint_{key}"):
            stage = st.selectbox("My learning stage", progress.STAGES,
                                index=progress.STAGES.index(record.get("stage", "Not started")))
            bookmark = st.checkbox("Bookmark this lesson", value=record.get("bookmark", False))
            note = st.text_area("Explain it in your own words. What breaks it?", value=record.get("note", ""), max_chars=4000)
            if st.form_submit_button("Save checkpoint"):
                progress.update_topic(key, stage=stage, bookmark=bookmark, note=note)
                st.success("Checkpoint saved for browser sync.")
        st.page_link("views/my_learning.py", label="My learning and notebook")
    after = curriculum.next_topic(key)
    if after:
        st.page_link(after.path, label=f"Next: {after.title}", icon=":material/arrow_forward:")
    query = urlencode({"template": "lesson-correction.md", "title": f"Lesson correction: {topic.title}",
                       "body": f"Lesson: {key}\n\nWhat seems incorrect or unclear?\n\nSuggested correction / primary source:\n"})
    st.link_button("Report an issue with this lesson", "https://github.com/gokcol/AI-LAB/issues/new?" + query)
