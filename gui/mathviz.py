"""Shared helpers for the numeric worked-example pages.

LaTeX builders that take real numpy arrays (so displayed matrices can never drift from
the arithmetic), plus the contribution bar chart used to show which terms drove a score.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def fmt(v, nd=2) -> str:
    return f"{float(v):.{nd}f}"


def bmat(A, nd=2) -> str:
    """numpy scalar / 1-D / 2-D array as a LaTeX bmatrix (1-D becomes a column vector)."""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    rows = A if A.ndim == 2 else A.reshape(-1, 1)
    body = r" \\ ".join(" & ".join(fmt(v, nd) for v in row) for row in rows)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def rowvec(A, nd=2) -> str:
    A = np.atleast_1d(np.asarray(A, dtype=float))
    return r"\begin{bmatrix}" + " & ".join(fmt(v, nd) for v in A) + r"\end{bmatrix}"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def set_state(values: dict):
    """Callback for preset buttons — write widget values before the rerun."""
    for k, v in values.items():
        st.session_state[k] = v


# --------------------------------------------------------------------------- #
# Activation functions — the one non-linear choice in a layer
# Each entry: (f, f', LaTeX, output range, one-liner, the honest detail)
# --------------------------------------------------------------------------- #
def _gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


ACTIVATIONS: dict[str, dict] = {
    "sigmoid": dict(
        f=lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        df=lambda z: (lambda s: s * (1 - s))(1 / (1 + np.exp(-np.clip(z, -500, 500)))),
        tex=r"\sigma(z)=\frac{1}{1+e^{-z}}", rng="(0, 1)",
        one="squashes anything into a probability",
        detail="**The probability squasher.** Maps the whole real line into $(0,1)$, so the output "
               "reads as *'how likely'* — and paired with cross-entropy it makes $z$ the exact "
               "**log-odds**. Its flaw is **saturation**: past $|z|\\approx5$ the curve is flat, so "
               "the gradient $\\sigma'=\\sigma(1-\\sigma)\\le0.25$ collapses toward 0 and learning "
               "stalls. Also not zero-centred (all outputs positive), which slows training. "
               "**Use it on a binary output layer — rarely in hidden layers today.**"),
    "tanh": dict(
        f=np.tanh, df=lambda z: 1 - np.tanh(z) ** 2,
        tex=r"\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}", rng="(−1, 1)",
        one="a zero-centred sigmoid",
        detail="**The zero-centred squasher.** Exactly a rescaled sigmoid "
               "($\\tanh z = 2\\sigma(2z)-1$) but centred on 0, so its outputs are as often "
               "negative as positive — which makes the next layer's job easier and training "
               "faster than with sigmoid. Its peak gradient is 1.0 (vs sigmoid's 0.25). Still "
               "**saturates** at both ends, so very deep stacks still suffer. **The classic "
               "hidden-layer choice before ReLU, and still standard inside RNNs/LSTMs.**"),
    "ReLU": dict(
        f=lambda z: np.maximum(0.0, z), df=lambda z: (z > 0).astype(float),
        tex=r"\mathrm{ReLU}(z)=\max(0,z)", rng="[0, ∞)",
        one="pass positives, block negatives",
        detail="**The modern default.** Dead simple and dead cheap: below threshold it outputs "
               "exactly **0** (the unit is *silent*), above it passes the value straight through "
               "with **gradient exactly 1** — so no saturation on the positive side and no "
               "vanishing gradient however deep the stack. That single property is what made very "
               "deep networks trainable. Two costs: the output isn't zero-centred, and a unit "
               "pushed permanently negative gets **gradient 0 forever** (a *dead unit*). **Use it "
               "in hidden layers unless you have a reason not to.**"),
    "Leaky ReLU": dict(
        f=lambda z: np.where(z > 0, z, 0.01 * z),
        df=lambda z: np.where(z > 0, 1.0, 0.01),
        tex=r"\mathrm{LReLU}(z)=\max(0.01z,\;z)", rng="(−∞, ∞)",
        one="ReLU that never fully dies",
        detail="**ReLU with a safety valve.** Instead of a hard 0, negatives get a small slope "
               "(0.01), so the gradient is never exactly zero and a unit can always recover — it "
               "fixes ReLU's **dead-unit** failure. In practice the win is usually modest, which "
               "is why plain ReLU remains the default; reach for this when you suspect many of "
               "your units have died."),
    "GELU": dict(
        f=_gelu,
        df=lambda z: (_gelu(z + 1e-4) - _gelu(z - 1e-4)) / 2e-4,
        tex=r"\mathrm{GELU}(z)=z\,\Phi(z)", rng="≈(−0.17, ∞)",
        one="a smooth ReLU — what Transformers use",
        detail="**The Transformer's activation.** Instead of ReLU's hard corner it weights the "
               "input by the probability $\\Phi(z)$ that a standard normal is below it — a "
               "*smooth*, slightly-dipping-below-zero version of ReLU. Being differentiable "
               "everywhere (no kink) suits very deep, attention-based stacks. **This is what runs "
               "inside GPT-family models** — including the Tiny GPT page."),
    "step": dict(
        f=lambda z: (z > 0).astype(float), df=lambda z: np.zeros_like(z),
        tex=r"\mathrm{step}(z)=\begin{cases}1&z>0\\0&z\le0\end{cases}", rng="{0, 1}",
        one="a hard yes/no — but untrainable",
        detail="**The original (1958) perceptron unit, and a dead end.** It gives a crisp binary "
               "decision, which is why it is perfect for the hand-built **logic gates** and the "
               "adder on *Neurons → computer*. But its derivative is **0 everywhere** (undefined "
               "at 0), so gradient descent gets no signal at all — **you cannot train a network "
               "of these with backprop.** Replacing it with the smooth sigmoid is precisely what "
               "unlocked learning."),
    "ELU": dict(
        f=lambda z: np.where(z > 0, z, np.exp(np.clip(z, -500, 0)) - 1),
        df=lambda z: np.where(z > 0, 1.0, np.exp(np.clip(z, -500, 0))),
        tex=r"\mathrm{ELU}(z)=\begin{cases}z&z>0\\ \alpha(e^{z}-1)&z\le0\end{cases}", rng="(−1, ∞)",
        one="ReLU with a smooth, saturating negative tail",
        detail="**Exponential Linear Unit.** Positive side is the identity (gradient 1, like "
               "ReLU); the negative side decays smoothly to **−1** instead of snapping to 0. "
               "Because it can output negatives, its activations are closer to **zero-centred**, "
               "which speeds training — and the saturation at −1 makes it robust to noisy inputs. "
               "Costs an `exp` on the negative side. A solid ReLU alternative, largely superseded "
               "by GELU/SiLU in modern architectures."),
    "SELU": dict(
        f=lambda z: 1.0507 * np.where(z > 0, z, 1.6733 * (np.exp(np.clip(z, -500, 0)) - 1)),
        df=lambda z: 1.0507 * np.where(z > 0, 1.0, 1.6733 * np.exp(np.clip(z, -500, 0))),
        tex=r"\mathrm{SELU}(z)=\lambda\,\mathrm{ELU}_{\alpha}(z)", rng="≈(−1.76, ∞)",
        one="a self-normalizing ELU",
        detail="**Scaled ELU.** ELU multiplied by carefully derived constants "
               "($\\lambda\\approx1.0507$, $\\alpha\\approx1.6733$) chosen so that activations "
               "**keep zero mean and unit variance as they propagate** — the network normalizes "
               "*itself*, no BatchNorm required. The catch: this guarantee only holds under strict "
               "conditions (LeCun-normal init, no ordinary dropout, plain feed-forward stacks), so "
               "it never became the default. Elegant theory, narrow applicability."),
    "SiLU / Swish": dict(
        f=lambda z: z / (1 + np.exp(-np.clip(z, -500, 500))),
        df=lambda z: (lambda s: s * (1 + z * (1 - s)))(1 / (1 + np.exp(-np.clip(z, -500, 500)))),
        tex=r"\mathrm{SiLU}(z)=z\cdot\sigma(z)", rng="≈(−0.28, ∞)",
        one="self-gated — the input times its own sigmoid",
        detail="**Sigmoid Linear Unit** (a.k.a. **Swish**, found by automated search at Google). "
               "The unit gates *itself*: multiply the input by $\\sigma(z)$, so large positives "
               "pass through and negatives are softly suppressed. Smooth, non-monotonic (dips to "
               "≈ −0.28), and consistently a hair better than ReLU on deep vision nets. Very close "
               "to GELU in shape and behaviour — **its gated cousin SwiGLU is used in Llama-family "
               "LLMs**."),
    "Softplus": dict(
        f=lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0),
        df=lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        tex=r"\mathrm{softplus}(z)=\ln(1+e^{z})", rng="(0, ∞)",
        one="a smooth ReLU — and its derivative is the sigmoid",
        detail="**The smooth approximation to ReLU.** No kink anywhere, and a lovely identity: "
               "its derivative is **exactly the sigmoid**, $\\frac{d}{dz}\\ln(1+e^z)=\\sigma(z)$. "
               "Because the output is strictly **positive**, it is the standard way to make a "
               "network emit a variance, a scale, or a rate that must stay > 0 (VAEs, "
               "probabilistic heads). Rarely used as a general hidden activation — it is more "
               "expensive than ReLU and never truly sparse (nothing is exactly 0)."),
    "Mish": dict(
        f=lambda z: z * np.tanh(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)),
        df=lambda z: ((lambda sp, t: t + z * (1 - t ** 2) / (1 + np.exp(-np.clip(z, -500, 500))))(
            np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0),
            np.tanh(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)))),
        tex=r"\mathrm{Mish}(z)=z\cdot\tanh(\mathrm{softplus}(z))", rng="≈(−0.31, ∞)",
        one="a smoother, deeper-dipping Swish",
        detail="**Mish** (2019) — the same self-gating idea as SiLU but gated by "
               "$\\tanh(\\mathrm{softplus}(z))$. Slightly smoother and dips a little lower "
               "(≈ −0.31). It reported small accuracy gains in object detection (it became "
               "popular via YOLOv4). Costs more to compute than ReLU for a modest, "
               "task-dependent gain — a good example of the diminishing returns in activation "
               "design after GELU/SiLU."),
    "hard sigmoid": dict(
        f=lambda z: np.clip(0.2 * z + 0.5, 0.0, 1.0),
        df=lambda z: np.where(np.abs(z) < 2.5, 0.2, 0.0),
        tex=r"\mathrm{hardsigmoid}(z)=\mathrm{clip}(0.2z+0.5,\,0,\,1)", rng="[0, 1]",
        one="a cheap piecewise-linear sigmoid",
        detail="**The mobile/edge approximation.** A straight line clipped to $[0,1]$ — no `exp`, "
               "just a multiply and two comparisons, which matters enormously on phones and "
               "microcontrollers and for **quantized** inference (Math **X6**). Used in "
               "MobileNetV3 (as hard-swish). The cost is a **flat gradient of 0 outside "
               "$|z|>2.5$** — a hard saturation that can stall learning, so it is an *inference* "
               "optimisation more than a training one."),
    "linear": dict(
        f=lambda z: z, df=lambda z: np.ones_like(z),
        tex=r"\mathrm{id}(z)=z", rng="(−∞, ∞)",
        one="no non-linearity at all",
        detail="**The identity — i.e. no activation.** Correct on a **regression output** layer, "
               "where you want an unbounded number (a price, a temperature). But **never** in a "
               "hidden layer: a stack of linear layers collapses, since "
               "$W^{(2)}(W^{(1)}\\mathbf x)=(W^{(2)}W^{(1)})\\mathbf x$ is just one matrix. Depth "
               "with linear activations buys **literally nothing** — this is why the "
               "non-linearity is non-negotiable (neuron lesson §8)."),
}


def activation_picker(key: str, default: str = "sigmoid", label: str = "activation φ"):
    """Segmented control over ACTIVATIONS; returns (name, spec dict)."""
    names = list(ACTIVATIONS)
    st.session_state.setdefault(key, default)
    name = st.segmented_control(label, names, key=key) or default
    return name, ACTIVATIONS[name]


def activation_explainer(name: str, z_mark: float | None = None):
    """Render the curve (+ derivative) and the written explanation for one activation."""
    spec = ACTIVATIONS[name]
    zz = np.linspace(-6, 6, 400)
    left, right = st.columns([0.46, 0.54])
    with left:
        fig, ax = plt.subplots(figsize=(4.3, 2.6))
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.plot(zz, spec["f"](zz), color="#185FA5", lw=2.2, label="φ(z)")
        ax.plot(zz, spec["df"](zz), color="#C0507A", lw=1.4, ls="--", label="φ′(z) — the gradient")
        if z_mark is not None and -6 <= z_mark <= 6:
            ax.plot([z_mark], [float(spec["f"](np.array([z_mark]))[0])], "o", ms=9,
                    color="#111827", zorder=5)
            ax.axvline(z_mark, color="#9C9B95", lw=1, ls=":")
        ax.set_xlabel("z", fontsize=9)
        ax.set_title(f"{name}   ·   range {spec['rng']}", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with right:
        st.latex(spec["tex"])
        st.markdown(f"**{name} — {spec['one']}.**  Output range **{spec['rng']}**.")
        st.markdown(spec["detail"])


def contribution_chart(labels, values, total, xlabel="contribution to z", height=2.6):
    """Horizontal bar chart of each term's contribution.

    Value labels are drawn *outside* the bar ends with the x-limits widened to match, so
    they can never overlap the bars or be clipped at the axis edge.
    """
    fig, ax = plt.subplots(figsize=(6.6, height))
    cols = ["#1D9E75" if v >= 0 else "#C0507A" for v in values]
    ax.barh(labels, values, color=cols, height=0.6)
    ax.axvline(0, color="0.35", lw=1.1)
    span = max(max(abs(float(v)) for v in values), 0.5)
    pad = span * 0.05
    ax.set_xlim(-span * 1.5, span * 1.5)
    for i, v in enumerate(values):
        v = float(v)
        ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.2f}", va="center", fontsize=9,
                ha="left" if v >= 0 else "right", color="#33312E")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.margins(y=0.14)
    ax.set_title(f"total = {total:+.3f}", fontsize=10)
    fig.tight_layout()
    return fig
