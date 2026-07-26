import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.OPTIMIZATION
START = np.array([-4.0, 4.0])

st.title("X4 · Optimization — playground & lesson")
st.caption("Watch gradient descent roll downhill on f(x,y) = x² + b·y². Push the learning rate "
           "until it diverges; raise the anisotropy to see it zig-zag.")

lessons.predict(
    'Push the **learning rate** up. What happens first — faster convergence or divergence? And what does raising the anisotropy (ravine steepness) do?',
    'Small η crawls; a larger η converges faster **up to a point**, then **overshoots and diverges**. High anisotropy turns the bowl into a ravine where gradient descent **zig-zags** — exactly why feature scaling (M7) and momentum / Adam matter.',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

LANDSCAPES = {
    "convex bowl  f = x²": (lambda x: x ** 2, lambda x: 2 * x),
    "two minima  f = x⁴ − 3x² + x": (lambda x: x ** 4 - 3 * x ** 2 + x,
                                     lambda x: 4 * x ** 3 - 6 * x + 1),
    "plateau  f = tanh²(x)": (lambda x: np.tanh(x) ** 2,
                              lambda x: 2 * np.tanh(x) * (1 - np.tanh(x) ** 2)),
    "wiggly  f = x² + 3sin(3x)": (lambda x: x ** 2 + 3 * np.sin(3 * x),
                                  lambda x: 2 * x + 9 * np.cos(3 * x)),
}


def _landscapes():
    st.markdown("**Not every loss is a bowl.** Real ones have local minima, plateaus and "
                "flat spots. Pick a landscape, choose where to start, and watch plain gradient "
                "descent — then switch **momentum** on and see what it changes.")
    c = st.columns(4)
    lname = c[0].selectbox("landscape", list(LANDSCAPES), index=1, key="o_land")
    x0 = c[1].slider("start x₀", -2.5, 2.5, 1.6, 0.1, key="o_x0")
    lr2 = c[2].slider("learning rate η", 0.001, 0.30, 0.05, 0.001, format="%.3f", key="o_lr2")
    mu = c[3].slider("momentum μ (0 = plain GD)", 0.0, 0.95, 0.0, 0.05, key="o_mu")
    nsteps = st.slider("steps", 5, 300, 80, 5, key="o_n2")

    f1, df1 = LANDSCAPES[lname]
    x, v = float(x0), 0.0
    xs_path, ls_path = [x], [float(f1(np.array([x]))[0])]
    for _ in range(int(nsteps)):
        g = float(np.atleast_1d(df1(np.array([x])))[0])
        v = mu * v + g                      # momentum accumulates past gradients
        x = x - lr2 * v
        if not np.isfinite(x) or abs(x) > 1e6:
            break
        xs_path.append(x); ls_path.append(float(np.atleast_1d(f1(np.array([x])))[0]))

    gg = np.linspace(-2.8, 2.8, 600)
    yy = np.atleast_1d(f1(gg))
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
    ax[0].plot(gg, yy, lw=2, color="#185FA5")
    ax[0].plot(xs_path, ls_path, "o-", ms=3.4, lw=1.2, color="#C0507A", alpha=.85, label="path")
    ax[0].plot([xs_path[0]], [ls_path[0]], "o", ms=10, color="#EF9F27", label="start", zorder=5)
    ax[0].plot([xs_path[-1]], [ls_path[-1]], "*", ms=17, color="#1D9E75", label="end", zorder=5)
    gmin = float(gg[int(np.argmin(yy))])
    ax[0].axvline(gmin, color="0.6", ls=":", lw=1)
    ax[0].set_title("landscape + descent path", fontsize=9.5); ax[0].set_xlabel("x", fontsize=9)
    ax[0].legend(fontsize=8); ax[0].tick_params(labelsize=8)
    ax[1].plot(ls_path, lw=2, color="#9A6A2A")
    ax[1].set_title("loss vs step", fontsize=9.5); ax[1].set_xlabel("step", fontsize=9)
    ax[1].tick_params(labelsize=8)
    fig.tight_layout(); st.pyplot(fig, width="stretch")

    xend = xs_path[-1]
    grad_end = abs(float(np.atleast_1d(df1(np.array([xend])))[0]))
    global_min = gmin
    at_global = abs(xend - global_min) < 0.25
    m = st.columns(4)
    m[0].metric("final x", f"{xend:+.4f}")
    m[1].metric("final f(x)", f"{ls_path[-1]:.4f}")
    m[2].metric("|gradient| at end", f"{grad_end:.2e}")
    m[3].metric("found", "🌍 global min" if at_global else "📍 local min / stuck")
    if not at_global and grad_end < 1e-2:
        st.warning(f"Converged to a **local** minimum at x ≈ {xend:.2f}, while the global one is "
                   f"near x ≈ {global_min:.2f}. Gradient descent is **local** — it only ever sees "
                   "the slope under its feet. Change the start, raise **momentum**, or add noise "
                   "(that is part of why **SGD** helps).", icon=":material/warning:")
    elif grad_end > 1e-2 and abs(xend) < 1e6:
        st.info("Still descending — the gradient is not yet ~0. On the **plateau** landscape the "
                "slope is nearly flat far from 0, so progress is glacial: that is the "
                "vanishing-gradient problem in one dimension.", icon=":material/timelapse:")
    else:
        st.success("Converged to the global minimum — gradient ≈ 0 at the bottom.",
                   icon=":material/check_circle:")
    st.info("**What momentum does:** $v \\leftarrow \\mu v + g$, then step by $-\\eta v$. It "
            "accumulates a running average of past gradients, so it **coasts through** small "
            "bumps and plateaus and damps oscillation. On *two minima*, start at x₀ ≈ 1.6 with "
            "μ = 0 to get trapped, then raise μ to ~0.9 and watch it roll over the hill.",
            icon=":material/speed:")


def _bowl():
    left, right = st.columns([0.42, 0.58])
    with left:
        lr = st.slider("learning rate η", 0.01, 1.10, 0.10, 0.01, key="o_lr")
        b = st.slider("anisotropy b (ravine)", 1.0, 20.0, 1.0, 1.0, key="o_b",
                      help="f = x² + b·y². Larger b → a steep, narrow ravine.")
        steps = st.slider("steps", 5, 100, 30, key="o_steps")
        normed = st.toggle("standardize the features (rescale y)", key="o_norm",
                           help="Change variable y' = √b·y so both axes have equal curvature — "
                                "exactly what feature scaling (M7) and normalization layers do.")
    # Standardizing rescales the flat axis so the ravine becomes a round bowl. In the new
    # coordinates y' = sqrt(b)*y the loss is x^2 + y'^2 — condition number 1.
    b_eff = 1.0 if normed else b

    def f(x, y):
        return x ** 2 + b_eff * y ** 2
    p = START.copy().astype(float)
    path = [p.copy()]
    losses = [float(f(*p))]
    diverged = False
    for _ in range(int(steps)):
        g = np.array([2 * p[0], 2 * b_eff * p[1]])  # gradient of f
        p = p - lr * g                              # GD update
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6:
            diverged = True
            break
        path.append(p.copy())
        losses.append(float(f(*p)))
    path = np.array(path)
    with right:
        g = np.linspace(-5, 5, 200)
        xx, yy = np.meshgrid(g, g)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.contour(xx, yy, f(xx, yy), levels=20, cmap="Blues", linewidths=0.8)
        vis = np.clip(path, -5, 5)
        ax.plot(vis[:, 0], vis[:, 1], "-o", color="#C0507A", ms=3, lw=1.3, label="GD path")
        ax.scatter([START[0]], [START[1]], color="#EF9F27", s=70, zorder=5, label="start")
        ax.scatter([0], [0], marker="*", color="#1D9E75", s=200, zorder=5, label="minimum")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title("loss surface + descent path")
        ax.legend(fontsize=8, loc="upper right")
        st.pyplot(fig, width="stretch")
    grew = losses[-1] > losses[0]
    if diverged or grew:
        status = "💥 diverging"
    elif losses[-1] < 0.05:
        status = "✅ converged"


with tab_play:
    omode = st.radio("playground",
                     ["2-D bowl & ravine", "🏔 1-D landscapes (local minima, momentum)"],
                     horizontal=True, key="o_mode")
    st.divider()
    if omode.startswith("2-D"):
        _bowl()
    else:
        _landscapes()


with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathopt")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** For a quadratic with largest Hessian eigenvalue $L$, GD is stable for $0<\eta<2/L$. Here $f=x^2+b y^2$ has $L=2\max(1,b)$, so the limit is $\eta<1/\max(1,b)$. With $b=1$ that is $\eta<1$; a tiny $\eta$ merely crawls.

**2.** High anisotropy makes a ravine, so the step is dominated by the steep axis → **zig-zag**. Feature scaling (M7) makes the axes comparable → a rounder bowl → straighter descent.""",
        label="Playground 1–2",
    )
    lessons.solution(
        r"""**3.** $L=w^2$, $\nabla=2w$, so $w\leftarrow w(1-2\eta)$. With $\eta=0.1,\,w_0=4$: $w_1=3.2,\ w_2=2.56,\ w_3=2.048$.

**4.** Converges when $|1-2\eta|<1\Rightarrow 0<\eta<1$. At $\eta=0.5$ it lands on the minimum in one step; $\eta>1$ diverges.""",
        label="Pencil & paper 3–4",
    )
    lessons.solution(
        r"""**5–6.** GD on $x^2+10y^2$ zig-zags down the $y$-ravine; adding **momentum** cuts the step count sharply.

**7.** Backprop computes $\nabla L$; this update changes the weights. Nets use **SGD** (mini-batches): noisier but far cheaper per step, and the noise helps escape shallow minima — which is why it beats full-batch at scale.""",
        label="Code & bridge 5–7",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
