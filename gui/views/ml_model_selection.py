import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons2

LESSON = ml_lessons2.MODEL_SELECTION
SCALE = 3.0


def _true(t):
    return np.sin(1.3 * t) + 0.3 * t


def _cv_errors(x, y, degree, k, seed):
    """k-fold CV: return (mean train MSE, mean validation MSE) for a polynomial of `degree`."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(x)), k)
    tr_e, val_e = [], []
    for i in range(k):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        d = min(degree, len(tr) - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = np.polyfit(x[tr] / SCALE, y[tr], d)
        pred = lambda t: np.polyval(coef, np.asarray(t) / SCALE)
        tr_e.append(np.mean((pred(x[tr]) - y[tr]) ** 2))
        val_e.append(np.mean((pred(x[val]) - y[val]) ** 2))
    return float(np.mean(tr_e)), float(np.mean(val_e))


st.title("M6 · Model selection — playground & lesson")
st.caption("Pick the model that wins on held-out data, not on training data. The validation "
           "curve (via k-fold CV) shows training error falling while CV error often forms a U.")

lessons.predict(
    'Training error always falls as models get more flexible. So why not just pick the lowest *training* error — and what curve should you use instead?',
    "Because lowest training error usually selects the most flexible model. Use **cross-validated** error: on noisy data it often forms a U, and its minimum estimates the best-generalizing complexity. The train–CV gap makes overfitting visible.",
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        n = st.slider("data points", 20, 200, 60, 10, key="ms_n")
        noise = st.slider("noise σ", 0.0, 1.0, 0.30, 0.05, key="ms_noise")
        k = st.slider("CV folds (k)", 3, 10, 5, key="ms_k")
        maxdeg = st.slider("max polynomial degree", 2, 15, 12, key="ms_maxdeg")
        seed = st.number_input("seed", 0, 9999, 0, key="ms_seed")

    rng = np.random.default_rng(int(seed))
    x = np.sort(rng.uniform(-3, 3, int(n)))
    y = _true(x) + rng.normal(0, noise, int(n))

    degs = list(range(1, int(maxdeg) + 1))
    tr_curve, val_curve = [], []
    for d in degs:
        te, ve = _cv_errors(x, y, d, int(k), int(seed))
        tr_curve.append(te)
        val_curve.append(ve)
    best_i = int(np.argmin(val_curve))
    best_deg = degs[best_i]

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.plot(degs, tr_curve, "-o", color="#5B8FC2", ms=3, label="train MSE")
        ax.plot(degs, val_curve, "-o", color="#C0507A", ms=3, label="CV MSE")
        ax.axvline(best_deg, ls="--", color="#1D9E75", label=f"best degree = {best_deg}")
        ax.set_yscale("log")
        ax.set_xlabel("model complexity (polynomial degree)")
        ax.set_ylabel("MSE (log scale)")
        ax.set_title(f"validation curve ({k}-fold CV)")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best degree (by CV)", best_deg)
    c2.metric("CV MSE @ best", f"{val_curve[best_i]:.3f}")
    c3.metric("Train MSE @ best", f"{tr_curve[best_i]:.3f}")
    st.info("**Train MSE generally falls** as flexibility grows, so selecting by it favors the "
            "most complex candidate. On noisy data **CV MSE often forms a U**; its minimum is "
            "the estimated best-generalizing model. With zero noise or finite-sample wiggles, "
            "the curve need not be perfectly U-shaped.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlmodelsel")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** 5-fold CV on 1000 rows trains **5** models; each uses **800** rows to train and **200** to validate (a different 200 per fold).

**2.** High-bias: train and validation error both plateau **high** and close together (underfit). High-variance: **low** train error but a **large gap** to validation error (overfit).

**3.** The **one-standard-error rule**: among models within 1 SE of the best CV score, pick the **simplest**. They're statistically indistinguishable from the best, so the simpler one is less likely to have overfit the CV noise.""",
        label="Pencil & paper 1–3",
    )
    lessons.solution(
        r"""**4–7.** Hand k-fold matches `cross_val_score`; putting `StandardScaler` **inside** a `Pipeline` scales per-fold and avoids the leakage of scaling-then-CV; `GridSearchCV` vs `RandomizedSearchCV` trade cost vs. coverage; validation & learning curves diagnose bias/variance.

**8.** Tuning against the test set means you *selected* using it, so the final score is optimistically biased — no longer an estimate of unseen performance. Tune on validation/CV; keep the test set untouched until the end.""",
        label="Code & concept 4–8",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
