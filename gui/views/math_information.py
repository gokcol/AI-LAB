import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.INFORMATION
EPS = 1e-3


def _H(a):
    terms = []
    if a > 0:
        terms.append(-a * np.log2(a))
    if a < 1:
        terms.append(-(1 - a) * np.log2(1 - a))
    return float(sum(terms))


st.title("X5 · Information theory — playground & lesson")
st.caption("Cross-entropy is the classification loss. Set the true p and the model's prediction "
           "q, and watch cross-entropy bottom out exactly when q = p (KL = 0).")

lessons.predict(
    "Set the model's prediction **q equal to the true p**. Where does cross-entropy bottom out, and what is KL there? What if q is confident but wrong?",
    'Cross-entropy is minimized exactly when **q = p**, where it equals the entropy H(p) and **KL = 0**. Being confident *and* wrong (q → 0 where p is high) makes it **explode** — that steep penalty is what trains classifiers (M2 / ANN).',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        p = st.slider("true P(class 1)  ·  p", 0.0, 1.0, 0.70, 0.01, key="i_p")
        q = st.slider("model prediction  ·  q", 0.0, 1.0, 0.50, 0.01, key="i_q")
    pc, qc = float(p), float(np.clip(q, EPS, 1 - EPS))
    Hp = _H(pc)
    CE = float(-pc * np.log2(qc) - (1 - pc) * np.log2(1 - qc))
    KL = CE - Hp

    with right:
        qs = np.linspace(0.01, 0.99, 200)
        ce = -pc * np.log2(qs) - (1 - pc) * np.log2(1 - qs)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.plot(qs, ce, color="#185FA5", lw=2, label="cross-entropy H(p,q)")
        ax.axvline(pc, ls="--", color="#1D9E75", lw=1.5, label="q = p (minimum)")
        ax.scatter([qc], [CE], color="#C0507A", s=70, zorder=5, label="your q")
        ax.set_xlabel("model prediction q")
        ax.set_ylabel("bits")
        ax.set_title("cross-entropy vs. prediction")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("entropy H(p)", f"{Hp:.3f}")
    c2.metric("cross-entropy H(p,q)", f"{CE:.3f}")
    c3.metric("KL(p‖q)", f"{KL:.3f}")
    st.info("Cross-entropy is minimized (= the entropy H(p), so **KL = 0**) exactly when **q = p**. "
            "Being confident and wrong (q → 0 while p is high) makes it **explode** — the same "
            "behavior that trains classifiers in M2 / ANN.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathinfo")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Fair coin: $H=1$ bit. Fair 4-sided die: $H=\log_2 4=2$ bits. Coin with $p=0.9$: $H=-0.9\log_2 0.9-0.1\log_2 0.1\approx0.47$ bits (less uncertainty).

**2.** $p=[1,0]$, $q=[0.7,0.3]$: $H(p,q)=-\sum p_i\log q_i=-\log(0.7)\approx0.357$ nats ($0.515$ bits).

**3.** $H(p,q)=-\sum p\log q=\underbrace{-\sum p\log p}_{H(p)}+\underbrace{\sum p\log\tfrac pq}_{D_{KL}(p\Vert q)}$. $H(p)$ is fixed by the data, so minimizing cross-entropy minimizes $D_{KL}$ — it drives $q\to p$.""",
        label="Pencil & paper 1–3",
    )
    lessons.solution(
        r"""**4–5.** Verify $D_{KL}(p\Vert q)\ge0$ with equality iff $p=q$; and the loss $-\log q$ for a true positive **explodes** as $q\to0$ (confident-and-wrong is punished hardest).

**6–7.** Cross-entropy here = the M2 logistic/softmax loss = ANN **e08** = MLE (X3 §7); and a decision tree's **information gain** (M3) is exactly entropy reduction.""",
        label="Code & bridge 4–7",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
