"""Optimizers compared (ANN module).

SGD vs Momentum vs Adam, racing down an ill-conditioned "ravine" loss surface — run on
the lab's own core.optim, on a 2-D quadratic whose gradients flow through core.engine.Value.
Watch how momentum and Adam handle the steep/flat mismatch that makes plain SGD zig-zag.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # repo root (core)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
from core.engine import Value
from core.optim import SGD, Adam, Momentum

START = (-4.5, -3.5)
OPTS = {"SGD": "#C0507A", "Momentum": "#5B8FC2", "Adam": "#1D9E75"}


def _make_opt(name, params, lr):
    if name == "Momentum":
        return Momentum(params, lr=lr, mu=0.9)
    if name == "Adam":
        return Adam(params, lr=lr)
    return SGD(params, lr=lr)


@st.cache_data(show_spinner=False)
def run(name: str, lr: float, steps: int, A: float):
    """Minimize f(x,y) = A·x² + y² with the real engine + optimizer; record the path."""
    x, y = Value(START[0]), Value(START[1])
    opt = _make_opt(name, [x, y], lr)
    xs, ys, losses = [x.data], [y.data], []
    for _ in range(steps):
        loss = A * x ** 2 + y ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        xs.append(x.data); ys.append(y.data)
        losses.append(float(A * x.data ** 2 + y.data ** 2))
        if abs(x.data) > 50 or abs(y.data) > 50:    # diverged — stop recording
            break
    return xs, ys, losses


_THEORY = r"""
## 1. What an optimizer does

Backprop hands you the gradient $\nabla L$ — the uphill direction for every weight. An
**optimizer** decides **how to step** with those gradients to drive the loss down. The base
move is gradient descent, $w \leftarrow w - \eta\,\nabla L$, but raw steps struggle on real
loss surfaces, so smarter rules (momentum, Adam) *shape* the step. This page races three on
the same surface.

## 2. Batch, stochastic & mini-batch

You can estimate the gradient on:
- the **whole dataset** (*batch* GD) — accurate but one step needs all the data → slow;
- **one example** (*stochastic* GD / SGD) — cheap and noisy, and the noise can help jump out
  of bad spots;
- a **mini-batch** of ~32–512 (the practical default) — best speed/accuracy trade-off and
  GPU-friendly.

"SGD" in practice means **mini-batch** SGD. That gradient noise is exactly why a real loss
curve is **jagged**, not a smooth glide.

## 3. The learning rate η — the #1 knob

$\eta$ scales every step. **Too large** → you overshoot and the loss **diverges** (flies
off); **too small** → it crawls and may never arrive. There's a stability ceiling set by the
surface's curvature (for a quadratic, roughly $\eta < 2/\text{(largest curvature)}$). It is
the single most impactful thing you tune — crank it in the demo to watch SGD blow up.

## 4. The hard case — ravines (ill-conditioning)

Real loss surfaces are **ill-conditioned**: far steeper in some directions than others (Math
**X4 §10**). Then $-\nabla L$ doesn't point at the minimum — it points mostly **across** a
long narrow valley. Plain GD **overshoots the steep direction and crawls the flat one**,
zig-zagging. The demo surface $f(x,y)=A x^2+y^2$ is exactly that: steep in $x$ (large $A$),
flat in $y$.

## 5. SGD — the baseline

$$ w \leftarrow w - \eta\,\nabla L. $$
Simple, memory-light, and with a good schedule still a top performer. Weaknesses: it
zig-zags in ravines and can **stall on plateaus and saddle points** — flat regions where
$\nabla L\approx0$, which are everywhere in high-dimensional nets.

## 6. Momentum — a heavy ball

Keep a **velocity** that exponentially averages recent gradients, and step with *that*:
$$ v \leftarrow \mu\,v + \nabla L,\qquad w \leftarrow w - \eta\,v. $$
Consistent directions (down the valley) **accumulate speed**; oscillating directions (across
the valley) **cancel**. So momentum rolls through ravines and coasts across plateaus — a
heavy ball with inertia. Typical $\mu=0.9$ (≈ averaging the last ~10 gradients);
**Nesterov** is a slightly look-ahead variant.

## 7. RMSProp — per-parameter scaling

Different weights can have wildly different gradient sizes. **RMSProp** divides each weight's
step by a running root-mean-square of *its own* recent gradients:
$$ s \leftarrow \rho\,s + (1-\rho)(\nabla L)^2,\qquad w \leftarrow w - \eta\,\frac{\nabla L}{\sqrt{s}+\varepsilon}. $$
A parameter with tiny gradients (the flat direction) still takes meaningful steps; one with
huge gradients (the steep direction) is reined in — directly attacking the ravine problem.

## 8. Adam = momentum + RMSProp

Adam combines both: a momentum-like mean $m$ **and** an RMSProp-like variance $v$, each an
exponential average, plus a **bias correction** (they start at 0, so early estimates are
scaled up):
$$ m\leftarrow\beta_1 m+(1-\beta_1)\nabla L,\qquad v\leftarrow\beta_2 v+(1-\beta_2)(\nabla L)^2, $$
$$ \hat m=\frac{m}{1-\beta_1^{\,t}},\quad \hat v=\frac{v}{1-\beta_2^{\,t}},\qquad w\leftarrow w-\eta\,\frac{\hat m}{\sqrt{\hat v}+\varepsilon}. $$
Dividing $\hat m$ by $\sqrt{\hat v}$ gives **each parameter its own effective step size** —
big where gradients are small, small where large — so Adam takes a near-direct route with
little tuning. Defaults $\beta_1=0.9,\ \beta_2=0.999,\ \eta\approx10^{-3}$. The **robust
default**, especially for transformers/LLMs. **AdamW** (decoupled weight decay) is the
common modern variant — it's what the e21 nanoGPT uses.

## 9. Schedules & weight decay

Two add-ons that pair with any optimizer:
- **Learning-rate schedule** — start higher, then **decay** (step / cosine), often after a
  short **warmup**: big steps early to make progress, small steps late to settle.
- **Weight decay** — gently shrink weights every step (L2 regularization) for better
  generalization (the **Regularization** page).

## 10. Which to use

- **Adam / AdamW** — the safe default; fast and forgiving. Start here.
- **SGD + momentum (+ a schedule)** — often **generalizes best** for large vision nets, at
  the cost of more tuning.
- Whatever you pick, **the learning rate is still the #1 knob** — in the demo, watch SGD
  diverge while Adam survives at the *same* $\eta$. *(Lab: `core/optim.py` implements SGD,
  Momentum and Adam.)*
"""

_QUIZ = [
    lessons.Question(
        "Why does plain SGD zig-zag on an ill-conditioned (ravine) surface?",
        ["the learning rate is always too small", "the gradient points across the valley, so it overshoots the steep direction and crawls the flat one",
         "it has no bias term", "momentum is too high"], 1,
        "When curvature differs a lot between directions, −∇L doesn't aim at the minimum, so GD bounces."),
    lessons.Question(
        "What does momentum add to SGD?",
        ["a per-parameter learning rate", "a velocity that averages recent gradients, building speed down consistent directions",
         "second derivatives", "weight decay"], 1,
        "v ← μv + ∇L; w ← w − ηv. Consistent gradients accumulate; oscillations cancel."),
    lessons.Question(
        "Adam's key idea beyond momentum is to…",
        ["remove the learning rate", "divide by a running estimate of each gradient's size, giving per-parameter step sizes",
         "use the exact Hessian", "train without gradients"], 1,
        "Dividing m̂ by √v̂ adapts the step per parameter — big where grads are small, small where they're large."),
    lessons.Question(
        "Across all three optimizers, the single most important knob is still…",
        ["the random seed", "the learning rate η", "the batch size", "the number of parameters"], 1,
        "Too large diverges, too small crawls — true for SGD, Momentum and Adam alike (X4 §3)."),
]

_TASKS = r"""
### In the Compare tab
1. At the default learning rate, compare the three **paths** — which zig-zags, which rolls
   straight down the valley? Which reaches the minimum in the fewest steps (loss curve)?
2. Increase **ravine steepness A** — SGD's zig-zag gets worse; do momentum/Adam still cope?
3. Crank the **learning rate** up until **SGD diverges** (flies off the plot), then note
   that Adam often still survives. Now make it tiny — everything crawls.

### Pencil & paper
4. Write the momentum update and explain why it cancels oscillations but accelerates a
   steady descent.
5. In Adam, what happens to the effective step in a direction whose gradients are
   consistently **tiny**? Why does that help in a ravine?

### Code / bridge
6. Swap the optimizer in the **MLP** page (SGD ↔ Adam) and compare how fast XOR is solved.
7. Connect to Math **X4** (gradient descent, conditioning) and `core/optim.py` (these exact
   updates).
"""

_REFS = r"""
- Sebastian Ruder — *An overview of gradient descent optimization algorithms* (the classic survey).
- Kingma & Ba (2015) — *Adam*. · Sutskever et al. (2013) — *On the importance of momentum*.
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 8 (optimization for training).
- distill.pub — *Why Momentum Really Works* (interactive intuition).
- In this lab: Math **X4** (optimization), the **MLP** page (optimizer dropdown), `core/optim.py`.
"""


st.title("Optimizers compared — SGD · Momentum · Adam")
st.caption("The same ravine, three optimizers, run on the lab's core.optim. Watch how each "
           "handles a surface that's steep one way and flat the other.")

lessons.predict(
    'On a **ravine** (steep one way, flat the other) at the *same* learning rate, which optimizer reaches the bottom fastest — SGD, Momentum, or Adam — and which one zig-zags?',
    'There is no optimizer-only winner: the shared **learning rate changes the race**. At the default η=0.2, SGD is above this ravine’s stability limit and diverges, while Adam usually finishes ahead of Momentum. Lower η to 0.1 and plain SGD wins this simple quadratic. The lesson is to compare paths **and tune η**, not declare a universal champion.',
)

tab_cmp, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🏁 Compare", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_cmp:
    st.markdown("Minimizing the **ravine** $f(x,y) = A\\,x^2 + y^2$ (steep in $x$, flat in "
                "$y$) from the same start. Tune the shared learning rate and the steepness:")
    c = st.columns(3)
    lr = c[0].select_slider("learning rate η", [0.02, 0.05, 0.1, 0.2, 0.3], value=0.2, key="opt_lr")
    A = c[1].select_slider("ravine steepness A", [2, 4, 6, 9, 12], value=6, key="opt_A")
    steps = c[2].select_slider("steps", [30, 60, 100, 150], value=100, key="opt_steps")

    runs = {name: run(name, lr, steps, float(A)) for name in OPTS}

    left, right = st.columns(2)
    with left:
        st.caption("Paths over the loss contours")
        fig, ax = plt.subplots(figsize=(4.3, 3.9))
        gx = np.linspace(-5, 5, 80); gy = np.linspace(-5, 5, 80)
        XX, YY = np.meshgrid(gx, gy)
        ax.contour(XX, YY, A * XX ** 2 + YY ** 2, levels=12, colors="#D9D8D1", linewidths=0.8)
        for name, col in OPTS.items():
            xs, ys, _ = runs[name]
            ax.plot(xs, ys, "-o", color=col, ms=2.6, lw=1.4, label=name)
        ax.scatter([0], [0], marker="*", s=180, color="#33312E", zorder=5, label="min")
        ax.scatter([START[0]], [START[1]], s=40, color="#33312E", zorder=5)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xlabel("x (steep)"); ax.set_ylabel("y (flat)")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
    with right:
        st.caption("Loss vs step (log scale)")
        fig2, ax2 = plt.subplots(figsize=(4.3, 3.9))
        for name, col in OPTS.items():
            _, _, losses = runs[name]
            ax2.semilogy(np.maximum(losses, 1e-9), color=col, lw=1.8, label=name)
        ax2.set_xlabel("step"); ax2.set_ylabel("loss (log)")
        ax2.legend(loc="upper right", fontsize=8)
        st.pyplot(fig2, width="stretch")

    cols = st.columns(3)
    for (name, col), cc in zip(OPTS.items(), cols):
        _, _, losses = runs[name]
        final = losses[-1] if losses else float("nan")
        diverged = abs(runs[name][0][-1]) > 50 or abs(runs[name][1][-1]) > 50
        cc.metric(name, "diverged" if diverged else f"{final:.2e}",
                  help="final loss after the chosen steps")
    st.info("All three are the lab's `core.optim` (SGD/Momentum/Adam) stepping on gradients "
            "from `core.engine.Value.backward()`. Same η: SGD zig-zags down the steep axis, "
            "Momentum rolls through, Adam adapts per-axis.", icon=":material/bolt:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="optim")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** At the default $\eta=0.2$, **SGD diverges** because the steep-axis stability limit is $1/A=1/6\approx0.167$; **Adam** normally finishes ahead of Momentum. At $\eta=0.1$, however, SGD converges extremely quickly on this deterministic quadratic and can beat both. The ranking is not universal: optimizer and learning rate must be evaluated together.

**2.** Higher steepness $A$ worsens SGD's zig-zag (the steep axis dominates). Momentum still copes (it damps the oscillation) and Adam copes best (it rescales each axis independently).

**3.** Crank the lr up and **SGD diverges** (flies off), while **Adam** often survives because its effective step shrinks where gradients are large. Tiny lr → everything crawls. Stability vs. speed is the whole trade-off.""",
        label="Compare tab 1–3",
    )
    lessons.solution(
        r"""**4.** Momentum: $v \leftarrow \mu v + g,\quad \theta \leftarrow \theta - \eta v$. On an oscillating axis the alternating-sign gradients partly **cancel** in $v$; on a steady axis they **accumulate**, so it damps oscillation *and* accelerates consistent descent.

**5.** Adam divides the step by $\sqrt{\hat v}$ (a running mean of $g^2$). A direction with consistently **tiny** gradients has a tiny denominator, so its effective step is **boosted** — the flat axis of a ravine no longer gets neglected, which is exactly what SGD struggles with.""",
        label="Pencil & paper 4–5",
    )
    lessons.solution(
        r"""**6.** Swapping SGD ↔ Adam on the **MLP** page: Adam usually solves XOR in fewer epochs at the same lr.

**7.** These are the exact updates in `core/optim.py`, and the ravine behavior is the **conditioning** story from Math **X4** — a poorly-conditioned (anisotropic) loss is what makes plain GD zig-zag.""",
        label="Code / bridge 6–7",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
