import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gui.resource_guard import limited_training

import lessons
import ml_lessons2

LESSON = ml_lessons2.SVM


def _kernel(A, B, kind, gamma):
    if kind == "linear":
        return A @ B.T
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


@limited_training
def _fit(K, y, C, lr=0.3, iters=500):
    """Kernel soft-margin classifier: min mean hinge + (λ/2) aᵀKa, via (sub)gradient descent."""
    n = len(y)
    a = np.zeros(n)
    b = 0.0
    lam = 1.0 / (C * n)
    for _ in range(iters):
        f = K @ a + b
        active = (y * f) < 1
        grad_a = -(K[:, active] @ y[active]) / n + lam * (K @ a)
        grad_b = -y[active].sum() / n
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


st.title("M4 · SVM & kernels — playground & lesson")
st.caption("A linear kernel draws a straight margin; the RBF kernel bends it. Try the linear "
           "kernel on XOR (it fails) then switch to RBF (it works).")

lessons.predict(
    'Try a **linear** kernel on XOR-like data. Will it separate the classes? What does switching to **RBF** change?',
    "The linear kernel **fails** (~50–75%) — XOR isn't linearly separable. **RBF** bends the boundary by implicitly lifting the data into a higher-dimensional space where it *is* separable. **C** trades margin width vs. errors; **γ** sets how local the boundary is (large γ → wiggly / overfit).",
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        data = st.selectbox("dataset", ["XOR (non-linear)", "blobs (linear)"], key="svm_data")
        kind = st.selectbox("kernel", ["RBF", "linear"], key="svm_kernel")
        C = st.slider("C (less ↔ more regularization)", 0.1, 10.0, 1.0, 0.1, key="svm_C")
        gamma = st.slider("γ (RBF reach)", 0.1, 5.0, 1.0, 0.1, key="svm_gamma",
                          disabled=(kind == "linear"))
        seed = st.number_input("seed", 0, 9999, 0, key="svm_seed")

    rng = np.random.default_rng(int(seed))
    if data.startswith("XOR"):
        X = rng.uniform(-3, 3, (120, 2))
        y = np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 1.0, -1.0)
        y *= np.where(rng.random(len(y)) < 0.08, -1.0, 1.0)   # a little label noise
    else:
        X = np.vstack([rng.normal([-1.6, -1.6], 1.0, (60, 2)),
                       rng.normal([1.6, 1.6], 1.0, (60, 2))])
        y = np.r_[-np.ones(60), np.ones(60)]

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    K = _kernel(Xs, Xs, kind.lower(), gamma)
    a, b = _fit(K, y, C)
    f_tr = K @ a + b
    acc = float((np.sign(f_tr) == y).mean())
    sv = (y * f_tr) <= 1.02            # points on/inside the margin = support vectors

    with right:
        lim = 3.4
        g = np.linspace(-lim, lim, 90)
        xx, yy = np.meshgrid(g, g)
        grid = (np.c_[xx.ravel(), yy.ravel()] - mu) / sd
        zz = (_kernel(grid, Xs, kind.lower(), gamma) @ a + b).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.7, vmin=-2, vmax=2)
        ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.8)            # boundary
        ax.contour(xx, yy, zz, levels=[-1, 1], colors="k", linewidths=0.7, linestyles="--")  # margins
        ax.scatter(X[y == -1, 0], X[y == -1, 1], s=16, color="#A32D2D", label="class −1")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color="#185FA5", label="class +1")
        ax.scatter(X[sv, 0], X[sv, 1], s=70, facecolors="none", edgecolors="k",
                   linewidths=1.0, label="support vectors")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{kind} kernel — boundary + margins")
        ax.legend(fontsize=7, loc="upper right")
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{acc:.2f}")
    c2.metric("Support vectors", int(sv.sum()))
    c3.metric("Kernel", kind)
    st.info("**Linear** kernel on **XOR** can't separate the classes (~50–75%). Switch to "
            "**RBF** and it bends the boundary to fit. **C** trades margin width vs. errors; "
            "**γ** sets how local the RBF boundary is (large γ → wiggly/overfit).",
            icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlsvm")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** The max-margin line is the one with the widest gap to the nearest points of each class; the **support vectors** are exactly those nearest points that touch the margin.

**2.** Moving a **support vector** shifts the boundary (they define it); moving a **far-away** point does nothing — only support vectors matter.

**3.** Large **C** punishes margin violations hard → a **narrow** margin that can overfit; small **C** tolerates violations → a **wider**, more regularized margin (can underfit).""",
        label="Pencil & paper 1–3",
    )
    lessons.solution(
        r"""**4–6.** Linear SVC fails on XOR while **RBF** bends to fit; sweeping **C** / **γ** trades margin width vs. wiggliness; **standardizing** features first matters a lot for RBF (it works on distances).

**7.** The optimization touches the data only through **dot products** $x_i\cdot x_j$. A kernel $K(x_i,x_j)$ returns that dot product in the mapped space *directly*, so you get a non-linear boundary **without** ever building the high-dimensional features — the kernel trick.

**8.** RBF-SVM assumes **smoothness/locality**, a tree assumes **axis-aligned** structure, logistic assumes **linear separability** (so it fails on XOR). Same data, different inductive biases.""",
        label="Code, concept & bridge 4–8",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
