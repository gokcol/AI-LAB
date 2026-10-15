import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import i18n
import lab
import ui
from gui import curriculum, progress, guided
from gui.guided_content import LESSONS, practice_summary

LANG = i18n.lang()
COPY = {
    "en": {
        "title": "Study Coach",
        "caption": "A quiet training loop: run, explain, derive, rebuild from memory.",
        "hero_title": "Today: make one idea yours",
        "hero_body": "Pick one experiment, run it, fill a checkpoint, answer five questions aloud, then break one thing on purpose.",
        "steps": [
            ("Run", "Run one experiment and copy the most surprising output into notes."),
            ("Explain", "Answer questions from memory before reading the code again."),
            ("Derive", "Rewrite one formula in your own words and name every symbol."),
            ("Break", "Change one assumption so the result fails; record the failure mode."),
        ],
        "tabs": ["Today", "Questions", "Self-checks", "Derivations", "Checkpoint"],
        "suggested_order": "Suggested order",
        "today": """
1. `e09_optimizers` — feel what the optimizer changes.
2. `e10_initialization` — see why starting weights matter.
3. `e11_regularization` — separate train loss from generalization.
4. `infobase/04_derivations/adam-update.md` — rewrite Adam from memory.
5. `learning/self_checks/tier2_self_check.md` — diagnose a failed training run.
        """,
        "callout": "Stop when you can explain the failure mode, not when the script passes.",
        "question_set": "Question set",
        "self_check": "Self-check",
        "derivation": "Derivation",
    },
    "tr": {
        "title": "Çalışma Koçu",
        "caption": "Sakin bir öğrenme döngüsü: çalıştır, açıkla, türet, hafızadan yeniden kur.",
        "hero_title": "Bugün: bir fikri gerçekten kendinin yap",
        "hero_body": "Bir deney seç, çalıştır, checkpoint doldur, beş soruyu sesli cevapla, sonra bilerek bir şeyi boz.",
        "steps": [
            ("Çalıştır", "Bir deneyi çalıştır ve en şaşırtıcı çıktıyı notlarına al."),
            ("Açıkla", "Koda tekrar bakmadan soruları hafızadan cevapla."),
            ("Türet", "Bir formülü kendi sözlerinle yeniden yaz ve her sembolü adlandır."),
            ("Boz", "Bir varsayımı değiştir; sonucun nasıl bozulduğunu kaydet."),
        ],
        "tabs": ["Bugün", "Sorular", "Öz-kontrol", "Türetmeler", "Checkpoint"],
        "suggested_order": "Önerilen sıra",
        "today": """
1. `e09_optimizers` — optimizer'ın neyi değiştirdiğini hisset.
2. `e10_initialization` — başlangıç ağırlıklarının neden önemli olduğunu gör.
3. `e11_regularization` — train loss ile genelleme arasındaki farkı ayır.
4. `infobase/04_derivations/adam-update.md` — Adam'ı hafızadan yeniden yaz.
5. `learning/self_checks/tier2_self_check.md` — başarısız bir eğitim koşusunu teşhis et.
        """,
        "callout": "Script geçince değil, hata modunu açıklayabildiğinde dur.",
        "question_set": "Soru seti",
        "self_check": "Öz-kontrol",
        "derivation": "Türetme",
    },
}
C = COPY.get(LANG, COPY["en"])

st.title(C["title"])
st.caption(C["caption"])

ui.hero(C["hero_title"], C["hero_body"])

cols = st.columns(4)
for col, (title, body) in zip(cols, C["steps"]):
    with col:
        st.markdown(f"<div class='ailab-step'><strong>{title}</strong><br>{body}</div>", unsafe_allow_html=True)

st.divider()

# Follow the sidebar on entry/change, while retaining an explicit coach choice on reruns.
module = st.session_state.get("module", "ANN")
if module not in ("ANN", "ML", "Math"):
    module = "ANN"
if st.session_state.get("coach_module_seen") != module:
    st.session_state["coach_track"] = module
    st.session_state["coach_module_seen"] = module
track = st.selectbox("Learning track", ["ANN", "ML", "Math"], key="coach_track")
focused = [key for key, lesson in LESSONS.items() if lesson["track"] == track]
st.caption(f"{track} companion · recommendations, questions, practice and derivations below follow this track.")

tab_today, tab_questions, tab_checks, tab_derivations, tab_template = st.tabs(C["tabs"])

with tab_today:
    st.subheader(C["suggested_order"])
    candidates = [t for t in curriculum.TOPICS if t.track == track]
    records = progress.state()["topics"]
    recommended = next((t for t in candidates
        if records.get(t.key, {}).get("stage") != "Rebuilt"
        and (t.key not in focused or practice_summary(t.key, progress.state().get("practice", {}))["independent"] < 8)), candidates[0])
    needs_practice = next((key for key in focused if practice_summary(key, progress.state().get("practice", {}))["independent"] < 8), None)
    if needs_practice:
        recommended = curriculum.CATALOG[needs_practice]
    st.page_link(recommended.path, label=f"Today: {recommended.title}")
    st.write(recommended.objective)
    st.caption(f"About {recommended.minutes} minutes. Guided topics use saved exercise results; other suggestions use your self-assessed checkpoints. You can skip ahead.")
    for key in focused:
        counts = practice_summary(key, progress.state().get("practice", {}))
        st.write(f"**{curriculum.CATALOG[key].title}:** {counts['solved']}/8 exercise types solved; {counts['independent']}/8 without hints or revealed answers.")
    if needs_practice:
        st.info("Begin with Guided start in the suggested lesson, or use Self-checks here. Reading a page does not mark its exercises solved.")
        if track == "ML":
            st.page_link("views/ml_foundations.py", label="New to ML? First explore features, targets and data splits")
    st.write("Predict one outcome, run the example, explain the result, then rebuild the smallest version without looking. Save a checkpoint at the bottom of the lesson.")
    st.page_link("views/my_learning.py", label="Review my notes and experiment notebook")
    explained = [t for t in candidates if records.get(t.key, {}).get("stage") in {"Explained", "Rebuilt"}]
    if explained:
        oldest = min(explained, key=lambda t: records[t.key].get("updated", ""))
        st.page_link(oldest.path, label=f"Recall practice: {oldest.title}")
        st.caption("Your least recently saved explained/rebuilt topic. Explain it again from memory before opening your notes.")
    st.markdown(
        f"<div class='ailab-callout'>{C['callout']}</div>",
        unsafe_allow_html=True,
    )

with tab_questions:
    if track == "ANN":
        qfiles = {
            "Tier 0": lab.LAB_ROOT / "learning/questions/tier0_neuron.md",
            "Tier 1": lab.LAB_ROOT / "learning/questions/tier1_backprop.md",
            "Tier 2": lab.LAB_ROOT / "learning/questions/tier2_training.md",
        }
        choice = st.segmented_control(
            C["question_set"],
            list(qfiles),
            default="Tier 2",
            key="study_coach_question_set",
        ) or "Tier 2"
        st.markdown(lab.read_localized(qfiles[choice], LANG), unsafe_allow_html=True)  # renders <details> answer keys
    else:
        st.subheader(f"{track} recall questions")
        st.caption("Answer aloud before opening the explanation. These reflections do not award checked practice results.")
        for key in focused:
            st.markdown("### " + LESSONS[key]["title"])
            for question, answer in LESSONS[key]["recall"]:
                st.write("**" + question + "**")
                with st.expander("Explanation · " + question):
                    st.write(answer)

with tab_checks:
    if track == "ANN":
        cfiles = {
            "Tier 0": lab.LAB_ROOT / "learning/self_checks/tier0_self_check.md",
            "Tier 1": lab.LAB_ROOT / "learning/self_checks/tier1_self_check.md",
            "Tier 2": lab.LAB_ROOT / "learning/self_checks/tier2_self_check.md",
        }
        choice = st.segmented_control(
            C["self_check"],
            list(cfiles),
            default="Tier 2",
            key="study_coach_self_check",
        ) or "Tier 2"
        st.markdown(lab.read_localized(cfiles[choice], LANG))
    else:
        chosen = st.selectbox("Practise a foundation", focused, format_func=lambda k: LESSONS[k]["title"], key="coach_practice_"+track)
        guided.render_exercises(chosen, namespace="coach")
        st.page_link(curriculum.CATALOG[chosen].path, label="Open the full guide and interactive example")

with tab_derivations:
    if track == "ANN":
        derivations = sorted(
            p for p in (lab.LAB_ROOT / "infobase/04_derivations").glob("*.md")
            if not lab.is_localized_variant(p)
        )
        labels = [p.stem.replace("-", " ") for p in derivations]
        selected = st.selectbox(C["derivation"], labels, index=0, key="study_coach_derivation")
        selected = selected or labels[0]
        path = derivations[labels.index(selected)]
        st.markdown(lab.read_localized(path, LANG))
    else:
        st.subheader(f"{track} · derive it one step at a time")
        for key in focused:
            st.markdown("### " + LESSONS[key]["title"])
            st.latex(LESSONS[key]["formula"])
            st.write(LESSONS[key]["symbols"])
            st.markdown(LESSONS[key]["derive"])
            with st.expander("Revisit the numerical example · " + curriculum.CATALOG[key].title):
                st.markdown(LESSONS[key]["worked"])

with tab_template:
    if track == "ANN":
        st.markdown(lab.read_localized(lab.LAB_ROOT / "learning/templates/checkpoint.md", LANG))
    else:
        st.subheader(f"{track} reflection")
        st.write("1. Explain one idea without a formula.\n2. Calculate one example without looking.\n3. Name a mistake you made and how you corrected it.\n4. Explain where this idea appears in the other track.")
        for key in focused:
            st.page_link(curriculum.CATALOG[key].path, label="Save a checkpoint in " + curriculum.CATALOG[key].title)
        st.page_link("views/my_learning.py", label="Review or export checkpoints, exercise results and notebook")
        st.caption("Checkpoints are your self-assessment. Checked exercise results are recorded separately and do not certify mastery.")
