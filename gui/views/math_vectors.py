import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons

LESSON = math_lessons.VECTORS

st.title("Linear algebra — vectors, matrices & transformations")
st.caption("X1 · The language of ML: a dot product is a neuron, a matrix is a layer, eigen/SVD is PCA. "
           "Two playgrounds below — the dot product, and a matrix transforming space.")

lessons.predict(
    'Drag **x** to point the *same* way as **w**, then perpendicular, then opposite. What does the dot product w·x do at each — and where is it largest?',
    "Largest (positive) when **aligned** (θ=0), **zero** when perpendicular (θ=90°), negative when opposed: w·x = ‖w‖‖x‖cos θ. That's exactly a neuron's pre-activation z = w·x + b — the neuron scores how much the input aligns with its weight direction.",
)

tab_dot, tab_matrix, tab_hd, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Dot product", "🔲 Matrix transform", "🌌 High dimensions", "📖 Theory",
     "❓ Self-check", "🛠 Tasks", "📚 References"]
)

# --------------------------------------------------------------------------- #
# Playground 1 — the dot product
# --------------------------------------------------------------------------- #
with tab_dot:
    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Vector w**")
        w1 = st.slider("w₁", -5.0, 5.0, 3.0, 0.1)
        w2 = st.slider("w₂", -5.0, 5.0, 1.0, 0.1)
        st.markdown("**Vector x**")
        x1 = st.slider("x₁", -5.0, 5.0, 1.0, 0.1)
        x2 = st.slider("x₂", -5.0, 5.0, 2.0, 0.1)

    w = np.array([w1, w2])
    x = np.array([x1, x2])
    dot = float(w @ x)
    nw, nx = float(np.linalg.norm(w)), float(np.linalg.norm(x))
    cos = max(-1.0, min(1.0, dot / (nw * nx))) if nw > 0 and nx > 0 else 0.0
    theta = float(np.degrees(np.arccos(cos))) if nw > 0 and nx > 0 else 0.0
    scalar_proj = dot / nw if nw > 0 else 0.0

    with right:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        lim = 5.6
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.annotate("", xy=(w1, w2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#C0507A", lw=2.5))
        ax.annotate("", xy=(x1, x2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1D6FB8", lw=2.5))
        ax.text(w1, w2, "  w", color="#C0507A", fontweight="bold")
        ax.text(x1, x2, "  x", color="#1D6FB8", fontweight="bold")
        if nw > 0:
            p = (dot / (nw * nw)) * w  # projection of x onto w
            ax.plot([p[0]], [p[1]], "o", color="#1D9E75")
            ax.plot([x1, p[0]], [x2, p[1]], "--", color="#1D9E75", lw=1)
            ax.text(p[0], p[1], "  proj", color="#1D9E75", fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("w·x = |w||x|cos θ")
        st.pyplot(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("w · x", f"{dot:.2f}")
    c2.metric("angle θ", f"{theta:.0f}°")
    c3.metric("cos θ", f"{cos:.2f}")
    c4.metric("proj of x on w", f"{scalar_proj:.2f}")
    st.latex(rf"w\cdot x = ({w1:.1f})({x1:.1f}) + ({w2:.1f})({x2:.1f}) = {dot:.2f}")
    st.info("This is exactly a neuron's pre-activation z = w·x (the bias just shifts it). "
            "Maximum when x points the same way as w (θ=0°); zero when perpendicular (θ=90°); "
            "negative when opposed.", icon=":material/lightbulb:")

# --------------------------------------------------------------------------- #
# Playground 2 — a matrix transforming space
# --------------------------------------------------------------------------- #
with tab_matrix:
    st.markdown("**Drag the columns of $W$** — they are exactly where the basis vectors "
                "$\\mathbf e_1,\\mathbf e_2$ land. Watch the unit square (dashed) map to a "
                "parallelogram, and see the determinant and eigenvectors update live.")
    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Column 1** — image of e₁ (blue)")
        a = st.slider("W₁₁  (x of We₁)", -3.0, 3.0, 1.0, 0.1, key="mt_a")
        c = st.slider("W₂₁  (y of We₁)", -3.0, 3.0, 0.5, 0.1, key="mt_c")
        st.markdown("**Column 2** — image of e₂ (coral)")
        b = st.slider("W₁₂  (x of We₂)", -3.0, 3.0, -0.6, 0.1, key="mt_b")
        d = st.slider("W₂₂  (y of We₂)", -3.0, 3.0, 1.2, 0.1, key="mt_d")
        show_eig = st.checkbox("show eigenvectors", value=True, key="mt_eig")
        show_vec = st.checkbox("show a test vector x → Wx", value=True, key="mt_vec")

    W = np.array([[a, b], [c, d]])
    det = float(a * d - b * c)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T  # 2×5
    tsquare = W @ square
    eig_msg = ""

    with right:
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        lim = 3.4
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.plot(square[0], square[1], color="#B9C4D0", lw=1.2, ls="--")
        ax.fill(tsquare[0], tsquare[1], color="#E6F1FB", alpha=0.55)
        ax.plot(tsquare[0], tsquare[1], color="#5B8FC2", lw=1.6)
        ax.annotate("", xy=(a, c), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#185FA5", lw=2.5))
        ax.annotate("", xy=(b, d), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#C0507A", lw=2.5))
        ax.text(a, c, "  We₁", color="#185FA5", fontsize=9, fontweight="bold")
        ax.text(b, d, "  We₂", color="#C0507A", fontsize=9, fontweight="bold")
        if show_vec:
            x0 = np.array([1.0, -0.8])
            wx0 = W @ x0
            ax.annotate("", xy=(x0[0], x0[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#9C9B95", lw=1.8))
            ax.annotate("", xy=(wx0[0], wx0[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#1D9E75", lw=2.2))
            ax.text(x0[0], x0[1], "  x", color="#6B6A66", fontsize=9)
            ax.text(wx0[0], wx0[1], "  Wx", color="#0E5E45", fontsize=9)
        if show_eig:
            vals, vecs = np.linalg.eig(W)
            if float(np.max(np.abs(vals.imag))) < 1e-9:
                for k in range(2):
                    ev = vecs[:, k].real
                    n = np.linalg.norm(ev)
                    if n > 0:
                        ev = ev / n * lim
                        ax.plot([-ev[0], ev[0]], [-ev[1], ev[1]],
                                color="#9A6A2A", lw=1.2, ls=":")
                eig_msg = f"real eigenvalues λ = {vals[0].real:.2f}, {vals[1].real:.2f} — dotted eigen-directions are only stretched"
            else:
                eig_msg = "eigenvalues are complex → W is a rotation, so there are no real eigen-directions"
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("W transforms the unit square")
        st.pyplot(fig, width="stretch")

    m = st.columns(3)
    m[0].metric("det W  (area ×)", f"{det:.2f}")
    m[1].metric("‖We₁‖", f"{np.hypot(a, c):.2f}")
    m[2].metric("‖We₂‖", f"{np.hypot(b, d):.2f}")
    if eig_msg:
        st.caption(eig_msg)
    st.info("The **columns of W** are where e₁ (blue) and e₂ (coral) land — together they map the "
            "whole square to a parallelogram, and stacking layers just multiplies these matrices. "
            "**det W** is the area factor: det = 0 collapses the square to a line (W is singular, "
            "non-invertible). Dotted lines are **eigen-directions** — vectors W only stretches, the "
            "basis of PCA.", icon=":material/lightbulb:")


# --------------------------------------------------------------------------- #
# Playground 3 — high-dimensional geometry (theory §17)
# --------------------------------------------------------------------------- #
with tab_hd:
    st.markdown("**Three things that are false in 2-D and true in 1000-D.** Move the "
                "dimension slider and watch each one arrive. Every number here is measured, "
                "not asserted — the theory tab (§17) gives the one-line reason for each.")
    hc = st.columns(3)
    d = hc[0].select_slider("dimension $d$", [2, 3, 5, 10, 30, 100, 300, 1000, 3000],
                            value=100, key="hd_d")
    npts = hc[1].slider("sample size", 100, 800, 300, 50, key="hd_n",
                        help="Points for the distance test; pairs for the angle test.")
    show = hc[2].radio("show", ["Distances", "Angles", "Volume"], key="hd_show",
                       horizontal=False)

    _rng = np.random.default_rng(0)
    if show == "Distances":
        X = _rng.normal(0, 1, (npts, d))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        v = D[np.triu_indices(npts, 1)]
        ratio = (v.max() - v.min()) / v.min()
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        ax.hist(v, bins=60, color="#2F7BEA", alpha=.85)
        ax.axvline(v.min(), color="#C0507A", lw=1.4, ls="--", label=f"min {v.min():.2f}")
        ax.axvline(v.max(), color="#1D9E75", lw=1.4, ls="--", label=f"max {v.max():.2f}")
        ax.set_xlabel("pairwise distance"); ax.set_ylabel("count")
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        m = st.columns(3)
        m[0].metric("closest pair", f"{v.min():.2f}")
        m[1].metric("farthest pair", f"{v.max():.2f}")
        m[2].metric("(max − min) / min", f"{ratio:.3f}")
        st.info(f"At $d={d}$ the farthest pair is only **{100*ratio:.1f}% farther** than the "
                f"closest. As $d$ grows this goes to zero like $1/\\sqrt d$ — the sum of $d$ "
                f"squared differences grows like $d$ while its spread grows like $\\sqrt d$. "
                f"This is the curse of dimensionality that weakens k-NN and k-means.",
                icon=":material/social_distance:")
    elif show == "Angles":
        A = _rng.normal(0, 1, (2000, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
        B = _rng.normal(0, 1, (2000, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)
        cos = (A * B).sum(1)
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        ax.hist(cos, bins=60, range=(-1, 1), color="#1D9E75", alpha=.85)
        ax.axvline(0, color="#33312E", lw=1)
        ax.set_xlabel("cos θ between two random unit vectors"); ax.set_ylabel("count")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        m = st.columns(3)
        m[0].metric("std of cos θ", f"{cos.std():.4f}")
        m[1].metric("predicted 1/√d", f"{1/np.sqrt(d):.4f}")
        m[2].metric("|cos θ| > 0.1", f"{(np.abs(cos) > 0.1).mean():.1%}")
        st.success(
            f"Measured **{cos.std():.4f}** against a prediction of **{1/np.sqrt(d):.4f}**. "
            f"Two unrelated directions in $d={d}$ score $0.00\\pm{cos.std():.3f}$, so a "
            f"cosine similarity of 0.4 sits **{0.4/max(cos.std(),1e-9):.0f} standard "
            f"deviations** from chance. That sharpness — not any blurring — is why cosine "
            f"similarity works on 1536-dimensional embeddings.",
            icon=":material/straighten:")
    else:
        ds = np.arange(1, 201)
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        for frac, col in ((0.10, "#C0507A"), (0.01, "#9A6A2A")):
            ax.plot(ds, 1 - (1 - frac) ** ds, lw=2, color=col,
                    label=f"outer {frac:.0%} of the radius")
        ax.axvline(d, color="#2F7BEA", ls="--", lw=1.4, label=f"$d={d}$")
        ax.set_xlabel("dimension $d$"); ax.set_ylabel("fraction of the ball's volume")
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        m = st.columns(3)
        m[0].metric("volume in the outer 10 %", f"{1 - 0.9 ** d:.4%}")
        m[1].metric("volume in the outer 1 %", f"{1 - 0.99 ** d:.2%}")
        m[2].metric("typical radius of a sample", f"√{d} ≈ {np.sqrt(d):.1f}")
        st.info(f"Scale a $d$-ball to 90 % of its radius and you keep $0.9^{{{d}}}$ of the "
                f"volume — at $d={d}$ that is {0.9 ** d:.2e}. The ball is a **hollow shell**. "
                f"The same fact means samples from $\\mathcal N(0, I_d)$ sit at radius "
                f"$\\approx\\sqrt d = {np.sqrt(d):.1f}$, nowhere near the mode at the "
                f"origin: *typical* and *most probable* are different things in high "
                f"dimension.", icon=":material/blur_circular:")


with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="vectors")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Same direction: $\theta=0°$, $\cos\theta=1$, dot product is **maximal** ($=\lVert\mathbf w\rVert\lVert\mathbf x\rVert$). Perpendicular: $\theta=90°$, $\cos\theta=0$, dot product $=0$.

**2.** Opposite: $\theta=180°$, $\cos\theta=-1$ → the dot product is **negative** (minimal).

**3.** Doubling $\lVert\mathbf x\rVert$ leaves $\cos\theta$ **unchanged** (same direction) but **doubles** $\mathbf w\cdot\mathbf x$ (it scales with length).""",
        label="Dot-product playground 1–3",
    )
    lessons.solution(
        r"""**4.** A **pure rotation** keeps both columns unit-length and perpendicular, so the area is unchanged: $\det W = 1$. Making the two columns **parallel** collapses the square to a line — the transform flattens 2-D onto 1-D, so $\det W = 0$ (singular, non-invertible).

**5.** An **eigenvector** lands on its **own dotted line** ($W\mathbf v=\lambda\mathbf v$ — same direction, only stretched by $\lambda$); a generic vector's $W\mathbf x$ points somewhere else (rotated). Diagonal or symmetric $W$ make the eigen-directions easy to spot.""",
        label="Matrix playground 4–5",
    )
    lessons.solution(
        r"""**6.** $\lVert(3,4)\rVert=\sqrt{9+16}=5$. $\cos\theta$ between $(1,0)$ and $(1,1)$ $=\dfrac{1}{1\cdot\sqrt2}=\dfrac{1}{\sqrt2}\approx0.707$ (45°).

**7.** $(1,2,3)\cdot(0,1,0)=0+2+0=2$. The one-hot vector zeroes every component except the 2nd, so the dot product **selects** that coordinate.

**8.** $\begin{bmatrix}1&-1\\2&0.5\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}1-2\\2+1\end{bmatrix}=\begin{bmatrix}-1\\3\end{bmatrix}$ — a $(2\times2)(2\times1)=(2\times1)$ product, matching §15's $W\mathbf x$.

**9.** A diagonal matrix scales each axis, so its **eigenvectors are the axes**: $\mathbf e_1=(1,0)$ with $\lambda=2$ and $\mathbf e_2=(0,1)$ with $\lambda=3$.

**10.** $\det W=0$ means $W$ collapses a dimension — it sends some nonzero vector to $\mathbf 0$, so the map isn't one-to-one and **nothing can undo it** (no inverse).

**11.** Least squares minimizes $\lVert\mathbf y-X\boldsymbol\beta\rVert^2$; setting the gradient to zero gives $X^\top(\mathbf y-X\boldsymbol\beta)=0$ — every **column of $X$ is orthogonal to the residual**, i.e. $\hat{\mathbf y}$ is $\mathbf y$'s projection onto $\text{span}(X)$.""",
        label="Pencil & paper 6–11",
    )
    lessons.solution(
        r"""**12–13.** `dot` / `norm` / `cosine_similarity` match `np.dot` / `np.linalg.norm`; and $\mathbf w\cdot\mathbf x=\lVert\mathbf w\rVert\lVert\mathbf x\rVert\cos\theta$ holds numerically for random vectors.

**14.** Center the cloud (subtract the mean), run `U, S, Vt = np.linalg.svd(Xc)`; the first row of `Vt` is **PC1** (the top principal axis). Plotting it through the mean reproduces the figure in §14.""",
        label="Code 12–14",
    )
    lessons.solution(
        r"""**15.** The neuron's $z=\mathbf w\cdot\mathbf x+b$ is exactly **one row** of $W\mathbf x+\mathbf b$, so the Playground's $z$ equals your hand dot product plus the bias. Stacking layers composes the transforms: $W_2(W_1\mathbf x)=(W_2W_1)\mathbf x$ — **composing = multiplying matrices** (§7).""",
        label="Bridge 15",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
