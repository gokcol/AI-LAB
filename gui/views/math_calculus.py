import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.CALCULUS
FUNCS = {
    "x²": (lambda x: x ** 2, lambda x: 2 * x, "2x"),
    "sin(x)": (np.sin, np.cos, "cos(x)"),
    "eˣ": (np.exp, np.exp, "eˣ"),
    "x³ − 3x": (lambda x: x ** 3 - 3 * x, lambda x: 3 * x ** 2 - 3, "3x² − 3"),
}

st.title("X2 · Calculus & gradients — playground & lesson")
st.caption("The derivative is the slope of the tangent — the limit of secant slopes as h→0. "
           "Shrink h and watch the secant become the tangent.")

lessons.predict(
    'Shrink the secant gap **h** toward 0. What does the secant slope approach — and how is that the same thing backprop relies on?',
    'It approaches the **tangent slope**, i.e. the analytic derivative f′(x); the error → 0 as h → 0. That limit *is* the derivative. Backprop is just the **chain rule** applied to these derivatives, composed layer by layer.',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

def _secant():
    left, right = st.columns([0.42, 0.58])
    with left:
        name = st.selectbox("function f(x)", list(FUNCS))
        x0 = st.slider("point x", -3.0, 3.0, 1.0, 0.1, key="c_x0")
        h = st.slider("secant gap h", 0.01, 2.0, 1.0, 0.01, key="c_h")
    f, df, dexpr = FUNCS[name]
    m_an = float(df(x0)); m_sec = float((f(x0 + h) - f(x0)) / h)
    with right:
        xs = np.linspace(-3.6, 3.6, 400); ys = f(xs)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.plot(xs, ys, color="#185FA5", lw=2, label="f(x)")
        tx = np.array([x0 - 1.7, x0 + 1.7])
        ax.plot(tx, f(x0) + m_an * (tx - x0), color="#1D9E75", lw=1.6, label="tangent (true slope)")
        ax.plot([x0, x0 + h], [f(x0), f(x0 + h)], "o--", color="#C0507A", lw=1.6,
                label="secant (slope ≈ f′)")
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_ylim(float(ys.min()) - 1, float(ys.max()) + 1)
        ax.set_title("derivative = slope of the tangent")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("f′(x) analytic", f"{m_an:.3f}")
    c2.metric("secant slope", f"{m_sec:.3f}")
    c3.metric("abs error", f"{abs(m_sec - m_an):.3f}")
    st.latex(rf"f'(x) = {dexpr} \quad\Rightarrow\quad f'({x0:.1f}) = {m_an:.3f}")
    st.info("As **h → 0** the secant slope converges to the analytic derivative (the error "
            "shrinks). That limit *is* the derivative — and the chain rule on it is backprop.",
            icon=":material/lightbulb:")


# --- chain rule: y = f(g(x)) ------------------------------------------------ #
_INNER = {
    "g(x) = 2x + 1": (lambda x: 2 * x + 1, lambda x: 2 * np.ones_like(x), "2"),
    "g(x) = x²":     (lambda x: x ** 2, lambda x: 2 * x, "2x"),
    "g(x) = sin(x)": (np.sin, np.cos, "cos(x)"),
}
_OUTER = {
    "f(u) = u²":     (lambda u: u ** 2, lambda u: 2 * u, "2u"),
    "f(u) = eᵘ":     (np.exp, np.exp, "eᵘ"),
    "f(u) = tanh(u)": (np.tanh, lambda u: 1 - np.tanh(u) ** 2, "1 − tanh²(u)"),
    "f(u) = sin(u)": (np.sin, np.cos, "cos(u)"),
}


def _chain():
    st.markdown("**The one rule backprop is built from.** To differentiate a *composition* "
                "$y=f(g(x))$ you multiply the two local slopes: how fast $u$ moves when $x$ moves, "
                "times how fast $y$ moves when $u$ moves.")
    st.latex(r"\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx} = f'(g(x))\cdot g'(x)")
    c = st.columns(3)
    gname = c[0].selectbox("inner g(x)", list(_INNER), key="c_g")
    fname = c[1].selectbox("outer f(u)", list(_OUTER), key="c_f")
    x0 = c[2].slider("point x", -2.0, 2.0, 0.8, 0.05, key="c_cx")
    g, dg, dgx = _INNER[gname]; f, df, dfu = _OUTER[fname]
    u0 = float(g(np.array([x0]))[0]) if hasattr(g(np.array([x0])), "__len__") else float(g(x0))
    y0 = float(f(np.array([u0]))[0]) if hasattr(f(np.array([u0])), "__len__") else float(f(u0))
    dudx = float(np.atleast_1d(dg(np.array([x0])))[0])
    dydu = float(np.atleast_1d(df(np.array([u0])))[0])
    dydx = dydu * dudx
    eps = 1e-6
    num = (float(np.atleast_1d(f(np.atleast_1d(g(np.array([x0 + eps])))))[0])
           - float(np.atleast_1d(f(np.atleast_1d(g(np.array([x0 - eps])))))[0])) / (2 * eps)

    m = st.columns(4)
    m[0].metric("u = g(x)", f"{u0:.4f}")
    m[1].metric("du/dx", f"{dudx:.4f}")
    m[2].metric("dy/du", f"{dydu:.4f}")
    m[3].metric("dy/dx = product", f"{dydx:.4f}")
    st.latex(rf"\frac{{dy}}{{dx}} = \underbrace{{{dydu:.4f}}}_{{f'(u)={dfu}}}"
             rf"\;\times\;\underbrace{{{dudx:.4f}}}_{{g'(x)={dgx}}}\;=\;{dydx:.4f}")
    st.latex(rf"\text{{numerical check (central difference)}}:\;{num:.4f}\quad"
             rf"\text{{error}}={abs(num-dydx):.2e}")

    xs = np.linspace(-2.4, 2.4, 400)
    us = np.atleast_1d(g(xs))
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax[0].plot(xs, us, color="#185FA5", lw=2)
    ax[0].plot([x0 - .7, x0 + .7], [u0 - .7 * dudx, u0 + .7 * dudx], color="#1D9E75", lw=1.6)
    ax[0].plot([x0], [u0], "o", color="#111827"); ax[0].set_title(f"u = g(x)   slope {dudx:.2f}", fontsize=9)
    ax[0].set_xlabel("x", fontsize=8)
    uu = np.linspace(float(us.min()) - .5, float(us.max()) + .5, 400)
    ax[1].plot(uu, np.atleast_1d(f(uu)), color="#C0507A", lw=2)
    ax[1].plot([u0 - .7, u0 + .7], [y0 - .7 * dydu, y0 + .7 * dydu], color="#1D9E75", lw=1.6)
    ax[1].plot([u0], [y0], "o", color="#111827"); ax[1].set_title(f"y = f(u)   slope {dydu:.2f}", fontsize=9)
    ax[1].set_xlabel("u", fontsize=8)
    ys = np.atleast_1d(f(np.atleast_1d(g(xs))))
    ax[2].plot(xs, ys, color="#9A6A2A", lw=2)
    ax[2].plot([x0 - .7, x0 + .7], [y0 - .7 * dydx, y0 + .7 * dydx], color="#1D9E75", lw=1.6)
    ax[2].plot([x0], [y0], "o", color="#111827")
    ax[2].set_title(f"y = f(g(x))   slope {dydx:.2f}", fontsize=9); ax[2].set_xlabel("x", fontsize=8)
    for a_ in ax:
        a_.tick_params(labelsize=8); a_.axhline(0, color="0.9", lw=.6)
    fig.tight_layout(); st.pyplot(fig, width="stretch")
    st.info("**This is exactly one backprop step.** A network is a long composition "
            "$L(\\varphi(W\\mathbf x))$; backprop walks it right-to-left multiplying these local "
            "slopes. The gradient reaching an early layer is a *product* of many such factors — "
            "which is why if each is < 1 it **vanishes**, and if each is > 1 it **explodes** "
            "(see the Activation-functions simulation).", icon=":material/link:")


# --- partial derivatives / gradient field ----------------------------------- #
_SURF = {
    "f = x² + y² (bowl)": (lambda X, Y: X ** 2 + Y ** 2,
                           lambda X, Y: (2 * X, 2 * Y), r"(2x,\;2y)"),
    "f = x² + 10y² (ravine)": (lambda X, Y: X ** 2 + 10 * Y ** 2,
                               lambda X, Y: (2 * X, 20 * Y), r"(2x,\;20y)"),
    "f = x² − y² (saddle)": (lambda X, Y: X ** 2 - Y ** 2,
                             lambda X, Y: (2 * X, -2 * Y), r"(2x,\;-2y)"),
    "f = sin(x)·cos(y)": (lambda X, Y: np.sin(X) * np.cos(Y),
                          lambda X, Y: (np.cos(X) * np.cos(Y), -np.sin(X) * np.sin(Y)),
                          r"(\cos x\cos y,\;-\sin x\sin y)"),
}


def _gradient_field():
    st.markdown("**With more than one input, the derivative becomes a vector.** Hold every other "
                "variable still and differentiate — that is a **partial derivative**. Collect them "
                "and you get the **gradient** $\\nabla f$, which points in the direction of "
                "*steepest ascent*.")
    c = st.columns(3)
    sname = c[0].selectbox("surface f(x, y)", list(_SURF), key="c_surf")
    px = c[1].slider("x", -2.5, 2.5, 1.2, 0.1, key="c_px")
    py = c[2].slider("y", -2.5, 2.5, 0.9, 0.1, key="c_py")
    f, grad, gexpr = _SURF[sname]
    gx, gy = grad(np.array([px]), np.array([py]))
    gx, gy = float(np.atleast_1d(gx)[0]), float(np.atleast_1d(gy)[0])
    norm = float(np.hypot(gx, gy))

    m = st.columns(4)
    m[0].metric("∂f/∂x", f"{gx:+.3f}")
    m[1].metric("∂f/∂y", f"{gy:+.3f}")
    m[2].metric("‖∇f‖ (steepness)", f"{norm:.3f}")
    m[3].metric("f(x, y)", f"{float(f(np.array([px]), np.array([py]))[0]):.3f}")
    st.latex(r"\nabla f = \left(\frac{\partial f}{\partial x},\; \frac{\partial f}{\partial y}\right) = "
             + gexpr + rf" = ({gx:.2f},\; {gy:.2f})")

    gg = np.linspace(-2.8, 2.8, 300)
    X, Y = np.meshgrid(gg, gg)
    Z = f(X, Y)
    qg = np.linspace(-2.5, 2.5, 13)
    QX, QY = np.meshgrid(qg, qg)
    U, V = grad(QX, QY)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cf = ax.contourf(X, Y, Z, levels=24, cmap="Blues", alpha=.75)
    ax.contour(X, Y, Z, levels=12, colors="0.55", linewidths=.7)
    ax.quiver(QX, QY, U, V, color="#6B6A66", alpha=.55, width=.003)
    if norm > 1e-9:
        ax.arrow(px, py, gx / norm * .9, gy / norm * .9, width=.05, color="#C0507A",
                 length_includes_head=True, zorder=6)
        ax.arrow(px, py, -gx / norm * .9, -gy / norm * .9, width=.05, color="#1D9E75",
                 length_includes_head=True, zorder=6)
    ax.plot([px], [py], "o", ms=9, color="#111827", zorder=7)
    ax.text(px + .12, py + .12, "  ∇f (uphill)", color="#C0507A", fontsize=8.5, zorder=7)
    ax.text(px - .12, py - .35, "  −∇f (downhill)", color="#1D9E75", fontsize=8.5, ha="right", zorder=7)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("contours + gradient field", fontsize=9.5)
    fig.colorbar(cf, ax=ax, shrink=.85)
    fig.tight_layout(); st.pyplot(fig, width="stretch")
    st.info("Two things to notice. **(1)** The gradient is always **perpendicular to the contour** "
            "— the fastest way uphill is straight across the level lines. **(2)** Gradient descent "
            "walks along the **green** arrow, $-\\nabla f$. On the *ravine* surface the arrows "
            "point mostly sideways rather than at the minimum — that is exactly the zig-zag you "
            "see in **X4**, and the saddle shows a point where $\\nabla f=0$ that is **not** a "
            "minimum.", icon=":material/explore:")


# --- gradient checking: the accuracy of a numerical derivative -------------- #
def _gradcheck():
    st.markdown("**How small should h be?** Smaller h means a better approximation — until "
                "floating-point round-off takes over and it gets *worse*. This is the "
                "trade-off behind gradient checking (experiment **e06**, Math **X6**).")
    c = st.columns(2)
    fname = c[0].selectbox("function", list(FUNCS), index=3, key="c_gc_f")
    x0 = c[1].slider("point x", -2.0, 2.0, 1.0, 0.1, key="c_gc_x")
    f, df, dexpr = FUNCS[fname]
    exact = float(df(x0))
    hs = np.logspace(-14, 0, 120)
    fwd = np.abs((f(x0 + hs) - f(x0)) / hs - exact)
    cen = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - exact)
    fwd = np.maximum(fwd, 1e-18); cen = np.maximum(cen, 1e-18)
    hb_f = float(hs[int(np.argmin(fwd))]); hb_c = float(hs[int(np.argmin(cen))])

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.loglog(hs, fwd, lw=2, color="#C0507A", label="forward  (f(x+h)−f(x))/h")
    ax.loglog(hs, cen, lw=2, color="#1D9E75", label="central  (f(x+h)−f(x−h))/2h")
    ax.axvline(hb_c, color="#1D9E75", ls=":", lw=1)
    ax.axvline(hb_f, color="#C0507A", ls=":", lw=1)
    ax.set_xlabel("step size h", fontsize=9); ax.set_ylabel("|error|", fontsize=9)
    ax.set_title("too big → truncation error · too small → round-off error", fontsize=9.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8); ax.grid(alpha=.25, which="both")
    fig.tight_layout(); st.pyplot(fig, width="stretch")

    m = st.columns(4)
    m[0].metric("exact f′(x)", f"{exact:.6f}")
    m[1].metric("best h · forward", f"{hb_f:.1e}")
    m[2].metric("best h · central", f"{hb_c:.1e}")
    m[3].metric("best error · central", f"{cen.min():.2e}")
    st.markdown(
        f"**Read the V.** Coming from the right, the error falls as $h$ shrinks — that is "
        f"**truncation** error, $O(h)$ for forward and $O(h^2)$ for central (hence the steeper "
        f"green line). Past the bottom it *rises* again: subtracting two nearly-equal numbers "
        f"loses precision, so **round-off** error $\\sim\\varepsilon/h$ blows up. The sweet spots "
        f"are around $\\sqrt{{\\varepsilon}}\\approx10^{{-8}}$ (forward) and "
        f"$\\varepsilon^{{1/3}}\\approx10^{{-5}}$ (central) — here **{hb_f:.0e}** and "
        f"**{hb_c:.0e}**.\n\n"
        f"Central differences are both **more accurate** and **more forgiving**, which is why "
        f"`e06` uses them to verify the autograd engine."
    )


with tab_play:
    mode = st.radio("playground",
                    ["Tangent & secant", "⛓ Chain rule (= backprop)",
                     "🧭 Partial derivatives & the gradient", "🔬 Gradient checking"],
                    horizontal=True, key="c_mode")
    st.divider()
    if mode.startswith("Tangent"):
        _secant()
    elif mode.startswith("⛓"):
        _chain()
    elif mode.startswith("🧭"):
        _gradient_field()
    else:
        _gradcheck()

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathcalc")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** $\frac{d}{dx}3x^4=12x^3$; $\frac{d}{dx}\sin x=\cos x$; $\frac{d}{dx}e^{2x}=2e^{2x}$; $\frac{d}{dx}(5x^2+2x-1)=10x+2$.

**2.** $\frac{d}{dx}(3x+1)^5=5(3x+1)^4\cdot3=15(3x+1)^4$; $\frac{d}{dx}e^{-x^2}=-2x\,e^{-x^2}$.

**3.** $\partial f/\partial x=2xy$, $\partial f/\partial y=x^2+3y^2$. At $(1,2)$: $\nabla f=(4,\,13)$.

**4.** $\nabla(x^2+y^2)=(2x,2y)=2(x,y)$ — at every point it points **radially outward** from the origin (steepest ascent of the bowl).""",
        label="Pencil & paper 1–4",
    )
    lessons.solution(
        r"""**5–6.** The central difference $\frac{f(x+h)-f(x-h)}{2h}$ matches $\frac{d}{dx}x^2=2x$ (the e06 gradient check); the tangent to $x^2$ at $x=2$ has slope 4.

**7.** Backprop *is* the chain rule: for a 2-layer net, $\partial L/\partial w^{(1)}$ is the product of local derivatives from the loss back through layer 2's activation and weights into layer 1 (see the **Backprop** page).""",
        label="Code & bridge 5–7",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
