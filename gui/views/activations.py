"""Activation functions — the nonlinearity, in depth (ANN module).

Compare every activation side by side, dive into the ReLU family (ReLU / Leaky ReLU /
GELU) with worked numbers, and run two real simulations that explain *why* the field
moved from sigmoid to ReLU: gradient flow through a deep stack, and dead units.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import mathviz as mv

st.title("Activation functions — the nonlinearity, in depth")
st.caption("The one non-linear choice in a layer. Compare them, dive into the ReLU family, "
           "and simulate why sigmoid lost to ReLU in deep networks.")

lessons.predict(
    "Stack **20 layers**. Which activation lets a gradient reach layer 1 roughly intact — and "
    "which shrinks it by a factor of about a *trillion*?",
    "**ReLU** passes it nearly intact (its derivative is exactly **1** wherever the unit is "
    "active). **Sigmoid** destroys it: its derivative never exceeds **0.25**, so 20 layers "
    "multiply the gradient by at most $0.25^{20}\\approx10^{-12}$. Tab ③ runs this for real.",
)

t1, t2, t3, t4, t5 = st.tabs(
    ["🎛 Compare", "⚡ The ReLU family", "🧪 Simulations", "🧭 Which one to use",
     "🚀 Beyond activations"])

# =========================================================================== #
# ① COMPARE
# =========================================================================== #
with t1:
    st.markdown(
        "An activation $\\varphi$ does one job: **bend the straight line**. Without it a network "
        "of any depth collapses to a single linear map (neuron lesson §8). Two curves matter for "
        "each one — the function $\\varphi(z)$ (what it outputs) and its **derivative** "
        "$\\varphi'(z)$ (how much gradient it lets through during backprop, Math **X2**)."
    )
    picks = st.multiselect("compare", list(mv.ACTIVATIONS),
                           default=["sigmoid", "tanh", "ReLU", "GELU"], key="act_cmp")
    zq = st.slider("evaluate at z =", -6.0, 6.0, 1.0, 0.1, key="act_z")

    if picks:
        zz = np.linspace(-6, 6, 500)
        colors = ["#185FA5", "#C0507A", "#1D9E75", "#9A6A2A", "#6C4FA1", "#2AA0B5", "#8A8A82"]
        fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
        for i, nm in enumerate(picks):
            s = mv.ACTIVATIONS[nm]
            c = colors[i % len(colors)]
            ax[0].plot(zz, np.asarray(s["f"](zz), dtype=float), lw=2.1, color=c, label=nm)
            ax[1].plot(zz, np.asarray(s["df"](zz), dtype=float), lw=2.1, color=c, label=nm)
        for a_, ttl in zip(ax, ["φ(z) — the output", "φ′(z) — the gradient that gets through"]):
            a_.axhline(0, color="0.85", lw=0.8)
            a_.axvline(0, color="0.85", lw=0.8)
            a_.axvline(zq, color="#111827", lw=1, ls=":")
            a_.set_title(ttl, fontsize=9.5)
            a_.set_xlabel("z", fontsize=9)
            a_.tick_params(labelsize=8)
            a_.legend(fontsize=7.5)
        ax[0].set_ylim(-1.6, 3.2)
        ax[1].set_ylim(-0.15, 1.3)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

        zv = np.array([zq])
        rows = []
        for nm in picks:
            s = mv.ACTIVATIONS[nm]
            fv = float(np.asarray(s["f"](zv), dtype=float)[0])
            dv = float(np.asarray(s["df"](zv), dtype=float)[0])
            rows.append(f"| **{nm}** | {s['rng']} | {fv:+.4f} | {dv:.4f} | "
                        + ("🟢 passes gradient" if dv > 0.5 else
                           ("🟡 weakens it" if dv > 0.01 else "🔴 blocks it")) + " |")
        st.markdown(f"**At z = {zq:.1f}:**\n\n"
                    "| activation | range | φ(z) | φ′(z) | gradient flow |\n|---|---|---|---|---|\n"
                    + "\n".join(rows))
        st.info("**Read the right-hand plot as 'how much learning signal survives this unit'.** "
                "Backprop multiplies by $\\varphi'(z)$ at every layer, so a curve that spends most "
                "of its range near **0** is a network that trains slowly or not at all.",
                icon=":material/lightbulb:")

# =========================================================================== #
# ② THE ReLU FAMILY
# =========================================================================== #
with t2:
    st.markdown(
        "Three relatives, in the order they were adopted. Each one keeps ReLU's key virtue — a "
        "**gradient of 1 on the positive side** — while fixing something about its behaviour on "
        "the negative side. *(The wider family — ELU, SELU, SiLU/Swish, Softplus, Mish, hard "
        "sigmoid — is in the **Compare** tab and the full table under **Which one to use**.)*"
    )
    fam = ["ReLU", "Leaky ReLU", "GELU"]
    zz = np.linspace(-4, 4, 500)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for i, nm in enumerate(fam):
        s = mv.ACTIVATIONS[nm]
        ax[i].axhline(0, color="0.85", lw=0.8)
        ax[i].axvline(0, color="0.85", lw=0.8)
        ax[i].plot(zz, np.asarray(s["f"](zz), dtype=float), lw=2.4, color="#185FA5")
        ax[i].plot(zz, np.asarray(s["df"](zz), dtype=float), lw=1.5, ls="--", color="#C0507A")
        ax[i].fill_between(zz, 0, np.asarray(s["f"](zz), dtype=float),
                           where=(zz < 0), color="#C0507A", alpha=0.14)
        ax[i].set_title(nm, fontsize=10)
        ax[i].set_xlabel("z", fontsize=9)
        ax[i].tick_params(labelsize=8)
    ax[0].set_ylim(-0.6, 3.0)
    ax[0].set_ylabel("blue φ(z) · dashed φ′(z)", fontsize=8.5)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.caption("The shaded strip is the negative region — the *only* place these three differ.")

    st.markdown("##### 🟢 ReLU — the rectifier")
    st.latex(r"\mathrm{ReLU}(z)=\max(0,z),\qquad \mathrm{ReLU}'(z)=\begin{cases}1&z>0\\0&z<0\end{cases}")
    st.markdown(
        "Named for the **rectifier** in electronics (a component that passes current one way "
        "only) — Hinton's *rectified linear neuron*. Below threshold the unit is **silent**; "
        "above it, it is a plain identity, so the gradient is **exactly 1** and nothing shrinks "
        "no matter how deep the stack. That is the property that made 20+ layer networks "
        "trainable, and why ReLU displaced sigmoid almost overnight after ~2011.\n\n"
        "- ✅ No saturation on the positive side → **no vanishing gradient**\n"
        "- ✅ Absurdly cheap (a comparison, no `exp`) and induces useful **sparsity** — typically "
        "about half the units output exactly 0\n"
        "- ❌ **Dead units**: if a unit's $z$ is negative for *every* input, its gradient is 0 "
        "forever and it can never recover (simulated in tab ③)\n"
        "- ❌ Output is never negative, so it is not zero-centred; the kink at 0 is "
        "non-differentiable (in practice frameworks just define $\\varphi'(0)=0$)"
    )

    st.markdown("##### 🟡 Leaky ReLU — the safety valve")
    st.latex(r"\mathrm{LReLU}(z)=\max(\alpha z,\;z)\ \ (\alpha\approx0.01),\qquad "
             r"\mathrm{LReLU}'(z)=\begin{cases}1&z>0\\ \alpha&z<0\end{cases}")
    st.markdown(
        "Identical to ReLU except the negative side has a **small slope** $\\alpha$ instead of a "
        "flat zero. That tiny leak means the gradient is **never exactly 0**, so a unit pushed "
        "negative can still be nudged back to life — it removes the dead-unit failure entirely. "
        "The cost is that the unit is no longer truly sparse (nothing is exactly 0). In practice "
        "the accuracy gain is usually small, which is why plain ReLU is still the default; reach "
        "for Leaky when you find a large fraction of your units dead. *(PReLU makes $\\alpha$ "
        "itself a learned parameter.)*"
    )

    st.markdown("##### 🔵 GELU — the smooth one that Transformers use")
    st.latex(r"\mathrm{GELU}(z)=z\cdot\Phi(z),\qquad \Phi(z)=P(\mathcal N(0,1)\le z)")
    st.markdown(
        "Instead of a hard on/off switch, GELU **weights the input by the probability that a "
        "standard normal falls below it** (Math **X3**). So it is a *soft* gate: at $z=0$ it "
        "passes half the input, at $z=2$ about 98 % of it. Consequences:\n\n"
        "- It is **smooth everywhere** — no kink at 0 — which suits very deep attention stacks "
        "trained with adaptive optimizers\n"
        "- It **dips slightly below zero** (minimum ≈ −0.17 near $z\\approx-0.75$), a small "
        "non-monotonic region that lets it represent 'mildly negative' evidence rather than "
        "discarding it\n"
        "- ❌ Costs more to compute than a comparison\n\n"
        "**This is the activation inside GPT-family models** — including the lab's own Tiny GPT "
        "and the e21 nanoGPT. *(SiLU/Swish, $z\\cdot\\sigma(z)$, is a close cousin.)*"
    )

    st.markdown("##### 🔢 Worked example — the same $z$ through all three")
    zs = [-2.0, -0.5, 0.0, 0.5, 2.0]
    hdr = "| z | ReLU | ReLU′ | LReLU | LReLU′ | GELU | GELU′ |\n|---|---|---|---|---|---|---|\n"
    body = []
    for zv in zs:
        za = np.array([zv])
        cells = []
        for nm in fam:
            s = mv.ACTIVATIONS[nm]
            cells += [f"{float(np.asarray(s['f'](za), dtype=float)[0]):+.4f}",
                      f"{float(np.asarray(s['df'](za), dtype=float)[0]):+.3f}"]
        body.append(f"| **{zv:+.1f}** | " + " | ".join(cells) + " |")
    st.markdown(hdr + "\n".join(body))
    st.markdown(
        "**Read the $z=-2$ row.** ReLU outputs 0 **and** hands back gradient 0 — that unit "
        "learned nothing from this example. Leaky ReLU outputs −0.02 with gradient 0.01: barely "
        "anything, but *not nothing*, so it can recover. GELU outputs −0.045 with a **negative** "
        "gradient — it is in its non-monotonic dip. And at $z=+2$ all three are essentially the "
        "identity with gradient ≈ 1: **they only differ where the evidence is negative.**"
    )
    st.latex(r"\text{Check GELU}(1)=1\cdot\Phi(1)=0.8413\ \ \text{(the tanh approximation gives }0.8412)")

# =========================================================================== #
# ③ SIMULATIONS
# =========================================================================== #
with t3:
    st.markdown("#### 🧪 Simulation 1 — how far does a gradient survive in a deep net?")
    st.markdown(
        "A real experiment, run live: build a random **L-layer** network of width $n$, do a "
        "forward pass, then backpropagate a random output gradient and record its **norm at "
        "every layer**. Weights use the right initializer for each activation (He for the ReLU "
        "family, Xavier for sigmoid/tanh) so the comparison is fair."
    )
    c = st.columns(3)
    L = c[0].slider("depth L (layers)", 5, 40, 20, 1, key="act_L")
    n = c[1].select_slider("width n", [16, 32, 64, 128], value=64, key="act_n")
    seed = c[2].number_input("seed", 0, 999, 0, 1, key="act_seed")
    sims = st.multiselect("activations to race", list(mv.ACTIVATIONS),
                          default=["sigmoid", "tanh", "ReLU", "GELU"], key="act_sims")
    normz = st.radio("insert a normalization layer?", ["none", "LayerNorm", "RMSNorm"],
                     horizontal=True, key="act_norm",
                     help="Normalizes each layer's pre-activations before φ. See the "
                          "'Beyond activations' tab for what this does and why.")

    def grad_norms(name, L, n, seed, norm="none"):
        """Forward + backward through a random deep net; returns |grad| at each layer.
        norm inserts LayerNorm / RMSNorm before the activation (exact backward)."""
        a = mv.ACTIVATIONS[name]
        rng = np.random.default_rng(int(seed))
        he = name in ("ReLU", "Leaky ReLU", "GELU", "SiLU / Swish", "ELU", "Mish")
        scale = np.sqrt(2.0 / n) if he else np.sqrt(1.0 / n)
        Ws = [rng.normal(0, scale, (n, n)) for _ in range(L)]
        h = rng.normal(0, 1, n)
        zs, hats, sds = [], [], []
        for W in Ws:
            z = W @ h
            if norm == "LayerNorm":
                sd = z.std() + 1e-5
                zh = (z - z.mean()) / sd
                hats.append(zh); sds.append(sd); z = zh
            elif norm == "RMSNorm":
                sd = np.sqrt((z ** 2).mean()) + 1e-5
                zh = z / sd
                hats.append(zh); sds.append(sd); z = zh
            zs.append(z)
            h = np.asarray(a["f"](z), dtype=float)
        g = rng.normal(0, 1, n)
        out = []
        for l in reversed(range(L)):
            g = g * np.asarray(a["df"](zs[l]), dtype=float)
            if norm == "LayerNorm":
                xh = hats[l]
                g = (g - g.mean() - xh * np.mean(g * xh)) / sds[l]
            elif norm == "RMSNorm":
                xh = hats[l]
                g = (g - xh * np.mean(g * xh)) / sds[l]
            out.append(float(np.linalg.norm(g)))
            g = Ws[l].T @ g
        return out[::-1]

    if sims:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        rows = []
        for i, nm in enumerate(sims):
            v = grad_norms(nm, L, n, seed, normz)
            ax.semilogy(range(1, L + 1), np.maximum(v, 1e-300), lw=2.1, marker="o", ms=3,
                        label=nm)
            ratio = v[-1] / v[0] if v[0] > 0 else float("inf")
            verdict = ("🔴 **vanished**" if ratio > 1e3 else
                       ("🟢 healthy" if 0.1 <= ratio <= 10 else "🟡 shrank"))
            rows.append(f"| **{nm}** | {v[0]:.3e} | {v[-1]:.3e} | {ratio:.2e}× | {verdict} |")
        ax.axhline(1.0, color="0.7", lw=1, ls=":")
        ax.set_xlabel("layer (1 = earliest / furthest from the loss)", fontsize=9)
        ax.set_ylabel("‖gradient‖ (log)", fontsize=9)
        ax.set_title(f"gradient reaching each layer · {L} layers · width {n}"
                     + ("" if normz == "none" else f" · with {normz}"), fontsize=9.5)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        st.markdown(
            "| activation | ‖grad‖ at layer 1 | at layer L | shrink factor | verdict |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows))
        st.info(
            "**This is the whole story of why ReLU won.** Sigmoid's derivative never exceeds "
            "**0.25**, so every layer multiplies the signal by at most a quarter — over 20 layers "
            "that is $0.25^{20}\\approx10^{-12}$, and the early layers receive essentially "
            "*nothing* (they never learn). ReLU's derivative is exactly **1** wherever the unit "
            "is active, so the signal is passed through undamped and the norm stays flat. Raise "
            "**L** and watch the sigmoid curve fall off a cliff.",
            icon=":material/trending_down:")

    st.divider()
    st.markdown("#### 🧪 Simulation 2 — dead ReLU units")
    st.markdown(
        "A ReLU unit is **dead** when its pre-activation is negative for *every* input in the "
        "data — it then outputs 0 always, receives gradient 0 always, and can never recover. "
        "The usual cause is a large negative bias (from a bad init or too big a learning-rate "
        "step). Drag the bias down and watch units die."
    )
    c2 = st.columns(2)
    bshift = c2[0].slider("bias applied to every unit", -4.0, 1.0, -2.0, 0.1, key="act_bshift")
    units = c2[1].select_slider("units in the layer", [50, 100, 200, 400], value=200,
                                key="act_units")
    rng = np.random.default_rng(1)
    Wd = rng.normal(0, 0.5, (int(units), 5))
    Xd = rng.normal(0, 1, (256, 5))
    Zd = Xd @ Wd.T + bshift
    dead_relu = float((Zd <= 0).all(axis=0).mean())
    active_frac = float((Zd > 0).mean())

    m = st.columns(3)
    m[0].metric("dead units · ReLU", f"{dead_relu*100:.1f}%",
                delta="permanently lost" if dead_relu > 0 else "none",
                delta_color="inverse" if dead_relu > 0 else "off")
    m[1].metric("dead units · Leaky ReLU", "0.0%", delta="always recoverable", delta_color="off")
    m[2].metric("average firing rate", f"{active_frac*100:.1f}%")

    fig2, ax2 = plt.subplots(figsize=(7.0, 2.6))
    shifts = np.linspace(-4, 1, 60)
    curve = [float((Xd @ Wd.T + s <= 0).all(axis=0).mean()) * 100 for s in shifts]
    ax2.plot(shifts, curve, lw=2.2, color="#C0507A", label="ReLU — dead units")
    ax2.axhline(0, color="#1D9E75", lw=2.2, ls="--", label="Leaky ReLU — never dead")
    ax2.axvline(bshift, color="#111827", lw=1, ls=":")
    ax2.set_xlabel("bias", fontsize=9)
    ax2.set_ylabel("% dead", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    st.warning(
        "Once a ReLU unit is dead its gradient is **exactly 0**, so gradient descent has no way "
        "to revive it — that slice of the network's capacity is gone for the rest of training. "
        "Defences: sensible **initialization** (He, experiment **e10**), a **learning rate** low "
        "enough not to slam biases negative, or simply **Leaky ReLU / GELU**, whose gradient is "
        "never exactly zero.", icon=":material/warning:")

# =========================================================================== #
# ④ WHICH ONE TO USE
# =========================================================================== #
with t4:
    st.markdown("#### 🎯 The short answer")
    st.markdown(
        "| where | use | why |\n|---|---|---|\n"
        "| **hidden layers** — default | **ReLU** | no vanishing gradient, cheapest, sparse |\n"
        "| hidden layers — many dead units | **Leaky ReLU** / **ELU** | gradient never exactly 0 |\n"
        "| **Transformers** / very deep | **GELU** or **SiLU** | smooth, best-in-class with Adam at depth |\n"
        "| **CNNs**, squeezing out accuracy | **SiLU** / **Mish** | small but real gains over ReLU |\n"
        "| **RNN / LSTM** internals | **tanh** + **sigmoid** gates | bounded state stops it blowing up |\n"
        "| **mobile / quantized** inference | **hard sigmoid** / hard-swish | no `exp`, integer-friendly |\n"
        "| output must be **positive** (a variance, a rate) | **softplus** | strictly > 0 and smooth |\n"
        "| **binary** output | **sigmoid** | calibrated probability; pairs with BCE |\n"
        "| **multi-class** output | **softmax** | a distribution; pairs with cross-entropy |\n"
        "| **regression** output | **linear** (none) | the answer is an unbounded number |\n"
        "| hand-built logic gates | **step** | crisp 0/1 — but **untrainable** |"
    )

    st.markdown("#### 📊 Every activation, side by side")
    props = {
        "sigmoid":      ("(0,1)",     "❌", "both ends", "0.25", "exp",      "—",     "output layer (binary)"),
        "tanh":         ("(−1,1)",    "✅", "both ends", "1.00", "exp",      "—",     "RNN / LSTM"),
        "ReLU":         ("[0,∞)",     "❌", "left only", "1.00", "compare",  "⚠️ yes", "hidden — the default"),
        "Leaky ReLU":   ("(−∞,∞)",    "~",  "no",        "1.00", "compare",  "no",    "hidden, if units die"),
        "GELU":         ("≈(−0.17,∞)", "~", "left only", "1.13", "exp/tanh", "no",    "Transformers / GPT"),
        "ELU":          ("(−1,∞)",    "~",  "left only", "1.00", "exp",      "no",    "hidden alternative"),
        "SELU":         ("≈(−1.76,∞)", "~", "left only", "1.76", "exp",      "no",    "self-normalizing nets"),
        "SiLU / Swish": ("≈(−0.28,∞)", "~", "left only", "1.10", "exp",      "no",    "CNNs, Llama (SwiGLU)"),
        "Softplus":     ("(0,∞)",     "❌", "left only", "1.00", "log/exp",  "no",    "positive outputs"),
        "Mish":         ("≈(−0.31,∞)", "~", "left only", "1.09", "log/exp",  "no",    "detection (YOLOv4)"),
        "hard sigmoid": ("[0,1]",     "❌", "both ends", "0.20", "compare",  "⚠️ yes", "mobile / quantized"),
        "step":         ("{0,1}",     "❌", "everywhere", "0.00", "compare", "n/a",   "logic gates only"),
        "linear":       ("(−∞,∞)",    "✅", "no",        "1.00", "none",     "no",    "regression output"),
    }
    rows = ["| activation | range | zero-centred | saturates | max φ′ | cost | dead units | typical home |",
            "|---|---|---|---|---|---|---|---|"]
    for nm, (rng, zc, sat, mg, cost, dead, home) in props.items():
        rows.append(f"| **{nm}** | {rng} | {zc} | {sat} | {mg} | {cost} | {dead} | {home} |")
    st.markdown("\n".join(rows))
    st.caption("*zero-centred:* ✅ symmetric about 0, ~ partially (can output small negatives), "
               "❌ output is one-signed. *max φ′* is the largest gradient the unit ever passes — "
               "anything well below 1 shrinks the signal at every layer.")

    st.markdown("#### 🕰 How the field actually got here")
    st.markdown(
        "1. **1958 · step** — the perceptron. Crisp, but gradient 0 → untrainable by backprop.\n"
        "2. **1980s–2000s · sigmoid, then tanh** — smooth and differentiable, which *enabled* "
        "backprop. tanh won over sigmoid for being zero-centred. Both **saturate**, so networks "
        "stayed shallow (2–3 layers): the vanishing gradient made deeper ones untrainable.\n"
        "3. **~2011 · ReLU** — the unlock. A gradient of exactly 1 on the positive side meant "
        "depth suddenly *worked*; AlexNet (2012) used it and the deep-learning era began.\n"
        "4. **2015 · Leaky ReLU / ELU / PReLU** — patching ReLU's dead-unit and "
        "not-zero-centred flaws.\n"
        "5. **2016–2018 · GELU, SiLU/Swish** — smooth, self-gating, slightly non-monotonic. "
        "Better suited to very deep stacks trained with Adam; **GELU became the Transformer "
        "standard**.\n"
        "6. **Now** — gains from new activations are *small*; the action moved to **gated "
        "variants** (SwiGLU in Llama) and to **normalization** (LayerNorm/RMSNorm), which "
        "arguably solved the conditioning problem more decisively than any activation."
    )

    st.markdown("#### 🩺 Troubleshooting — symptom → suspect → fix")
    st.markdown(
        "| symptom | likely cause | what to try |\n|---|---|---|\n"
        "| loss barely moves; early layers frozen | **vanishing gradient** (sigmoid/tanh deep) | switch hidden layers to **ReLU/GELU**; add **normalization**; check init |\n"
        "| loss → NaN or explodes | exploding gradient / η too high | lower **η**, **gradient clipping**, check init scale |\n"
        "| many units always output 0 | **dead ReLU** | **Leaky ReLU / ELU / GELU**, lower η, **He** init (e10) |\n"
        "| trains, but slower than expected | activations not zero-centred | **tanh / ELU / GELU**, or normalize activations |\n"
        "| output is a probability but unbounded | wrong **output** activation | **sigmoid** (binary) or **softmax** (multi-class) |\n"
        "| model outputs negative variances/rates | unconstrained output | **softplus** (or exp) on that head |\n"
        "| fine in training, too slow on device | `exp` cost | **hard sigmoid / hard-swish**, quantize (X6) |"
    )

    st.markdown("#### 🧠 Rules of thumb worth memorising")
    st.markdown(
        "- **The output activation is decided by the task, not by taste** — probability → "
        "sigmoid; classes → softmax; a number → nothing at all.\n"
        "- **The hidden activation is decided by depth and budget** — ReLU unless you are "
        "training a Transformer (GELU/SiLU) or fighting dead units (Leaky/ELU).\n"
        "- **Never put a linear activation in a hidden layer** — the network collapses to one "
        "layer (proved live in *Worked examples II*, tab ④).\n"
        "- **Never mix a saturating hidden activation with great depth** unless you also "
        "normalize.\n"
        "- **If you are unsure, ReLU is never a disaster** — the difference between good "
        "activations is a percent or two; the difference between a good one and sigmoid-at-depth "
        "is whether the model trains at all."
    )

    st.markdown("#### 🔗 Where this connects")
    st.markdown(
        "- **Math X2 — calculus:** every $\\varphi'(z)$ here is a derivative, and backprop is the "
        "chain rule multiplying them layer by layer (see the X2 *Chain rule* playground).\n"
        "- **Math X5 — information theory:** softmax + cross-entropy is why a sigmoid output "
        "makes $z$ the log-odds.\n"
        "- **Math X6 — numerics:** naive `exp` overflows, so sigmoid/softmax use the log-sum-exp "
        "trick (`core/activations.py`); hard variants exist for quantized inference.\n"
        "- **Neuron lesson §5–§8** — activations in the single-neuron setting, and why the "
        "nonlinearity is non-negotiable.\n"
        "- **e07 activation zoo · e10 initialization** — the same comparisons on the lab's own "
        "engine, and the init that keeps ReLU units alive."
    )
    st.info("**One rule if you remember nothing else:** *ReLU in the hidden layers, and let the "
            "**task** pick the output activation* — sigmoid for a probability, softmax for "
            "classes, nothing at all for a number.", icon=":material/lightbulb:")

# =========================================================================== #
# ⑤ BEYOND ACTIVATIONS — GATING & NORMALIZATION
# =========================================================================== #
with t5:
    st.markdown(
        "Activation research has **plateaued**: the gap between ReLU, GELU, SiLU and Mish is a "
        "percent or two. The two ideas that actually moved modern architectures forward are "
        "**gating** (a smarter way to *use* an activation) and **normalization** (fixing the "
        "conditioning of the whole layer). Both are, in a sense, admissions that the activation "
        "alone was never the bottleneck."
    )

    st.markdown("### 🚪 Gated units — GLU and SwiGLU")
    st.markdown(
        "A plain feed-forward layer applies one activation to one projection. A **gated linear "
        "unit** splits the input into **two** projections and lets one *multiplicatively gate* "
        "the other:"
    )
    st.latex(r"\mathrm{GLU}(\mathbf x)=\big(W_1\mathbf x\big)\;\odot\;\sigma\big(W_2\mathbf x\big)")
    st.latex(r"\mathrm{SwiGLU}(\mathbf x)=\big(W_1\mathbf x\big)\;\odot\;\mathrm{SiLU}\big(W_2\mathbf x\big)")
    st.markdown(
        "The second branch produces a **per-dimension gate in charge of how much of the first "
        "branch survives** — the network learns *what to let through*, rather than applying one "
        "fixed curve to everything. Notice this is the same self-gating trick as SiLU "
        "($z\\cdot\\sigma(z)$), except the gate is computed from a **separate learned "
        "projection** instead of from $z$ itself.\n\n"
        "- ✅ Consistently better perplexity per parameter in Transformer feed-forward blocks\n"
        "- ✅ **SwiGLU is what Llama, PaLM and Mistral use**; GeGLU (GELU-gated) is the same idea\n"
        "- ❌ Two projections instead of one, so implementations shrink the hidden width "
        "(often to $\\tfrac23$) to keep the parameter count fair\n\n"
        "*The lesson: the win came from **structure** (multiplicative interaction), not from a "
        "better-shaped curve.*"
    )

    st.markdown("### 🏭 What do the frontier models actually use?")
    st.markdown(
        "Worth separating two things: what is **published**, and what is **inferred**. Model "
        "architecture is a competitive asset, so the labs behind ChatGPT, Gemini and Claude "
        "**do not disclose** their current activation or normalization choices. What we *do* have "
        "is (a) their own older papers, and (b) the **open-weight** models from the same labs and "
        "their peers — which is plenty to see the trend."
    )
    st.markdown(
        "| model | activation | normalization | how we know |\n|---|---|---|---|\n"
        "| **GPT-2** (OpenAI, 2019) | **GELU** | LayerNorm, pre-norm | paper + open weights |\n"
        "| **GPT-3** (OpenAI, 2020) | **GELU** | LayerNorm, pre-norm | paper |\n"
        "| **GPT-4 / GPT-5** — today's *ChatGPT* | *not published* | *not published* | — |\n"
        "| **PaLM** (Google, 2022) | **SwiGLU** | RMSNorm, parallel blocks | paper |\n"
        "| **Gemma** (Google, open) | **GeGLU** | RMSNorm | model card + open weights |\n"
        "| **Gemini** (Google) | *not published* | *not published* | — |\n"
        "| **Llama 1–3** (Meta, open) | **SwiGLU** | RMSNorm, pre-norm | paper + open weights |\n"
        "| **Mistral / Mixtral** (open) | **SwiGLU** | RMSNorm | open weights |\n"
        "| **Qwen, DeepSeek** (open) | **SwiGLU** | RMSNorm | open weights |\n"
        "| **Claude** (Anthropic) | *not published* | *not published* | — |"
    )
    st.warning(
        "**Be sceptical of confident claims here.** You will find blog posts asserting exactly "
        "which activation GPT-4, Gemini or Claude uses — those are **guesses**, usually "
        "extrapolated from the open models in the same row. Anthropic, OpenAI and Google DeepMind "
        "publish capabilities and safety work, not layer-level configs. *(This lab is written by "
        "Claude, and no — that does not come with verified knowledge of Claude's own internals.)*",
        icon=":material/help:")
    st.markdown(
        "**But the pattern in the open column is unambiguous**, and it is very likely the closed "
        "models sit in the same place:\n\n"
        "1. **2018–2021 — the GELU era.** BERT, GPT-2, GPT-3: a plain feed-forward block with "
        "**GELU**, wrapped in **LayerNorm**.\n"
        "2. **2022 — gating arrives.** PaLM adopts **SwiGLU**; the community follows once the "
        "Llama weights ship with it. Essentially every open model released since is "
        "**SwiGLU or GeGLU**.\n"
        "3. **Simultaneously — LayerNorm → RMSNorm.** Drop the mean-subtraction, keep the scale; "
        "same benefit, cheaper at scale. Now near-universal in LLMs.\n"
        "4. **Pre-norm everywhere**, for the clean residual gradient highway.\n\n"
        "So the honest answer to *'what does ChatGPT/Gemini/Claude use?'* is: **the labs have not "
        "said** — but every comparable open model converged on **a gated GLU variant + RMSNorm + "
        "pre-norm**, and there is no public evidence the frontier models diverge from it."
    )
    st.info(
        "**The transferable lesson** — notice what the industry *stopped* arguing about. Nobody "
        "ships a new hand-designed curve any more; the last decade's real wins were **ReLU** "
        "(made depth trainable), **normalization** (made it stable), and **gating** (made the "
        "feed-forward block smarter). Your own models should just inherit that consensus: "
        "**GELU/SiLU, or SwiGLU if you are building a Transformer, with RMSNorm and pre-norm.**",
        icon=":material/lightbulb:")

    st.markdown("### 📏 Normalization — BatchNorm, LayerNorm, RMSNorm")
    st.markdown(
        "Normalization rescales a layer's activations to a controlled mean/variance **before** "
        "the next layer sees them. It is the single most effective fix for the deep-network "
        "conditioning problem — arguably more decisive than any activation."
    )
    st.latex(r"\hat z=\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\hat z+\beta")
    st.markdown(
        "| | normalizes over | needs a batch? | typical home |\n|---|---|---|---|\n"
        "| **BatchNorm** (2015) | the **batch**, per feature | ✅ yes | CNNs / vision |\n"
        "| **LayerNorm** (2016) | the **features**, per sample | ❌ no | Transformers, RNNs |\n"
        "| **RMSNorm** (2019) | features, **scale only** (no mean subtraction) | ❌ no | Llama, modern LLMs |\n\n"
        "**Why LayerNorm won in Transformers.** BatchNorm's statistics depend on the *other "
        "samples in the batch*, which breaks with variable-length sequences, tiny batches, and "
        "autoregressive inference (at generation time there is no batch). LayerNorm normalizes "
        "**within one sample**, so it behaves identically in training and inference.\n\n"
        "**Why RMSNorm then won again.** It drops the mean-subtraction and the $\\beta$ shift, "
        "keeping only the scale: $\\hat z = z/\\mathrm{RMS}(z)$. Nearly the same benefit, "
        "measurably cheaper — and at LLM scale, cheap wins.\n\n"
        "**Pre-norm vs post-norm.** The original Transformer normalized *after* the residual "
        "block; every modern one normalizes *before* it (**pre-norm**), because that leaves a "
        "clean, unnormalized residual highway for gradients to flow down — which is what makes "
        "100+ layer stacks trainable without warmup gymnastics."
    )

    st.markdown("### 🧪 Does normalization actually rescue a bad activation?")
    st.markdown("A live re-run of the depth simulation — same net, same seed, **with and without** "
                "LayerNorm. This is the claim, tested:")
    cN = st.columns(3)
    nact = cN[0].selectbox("activation", list(mv.ACTIVATIONS), index=0, key="t5_act")
    nL = cN[1].slider("depth", 5, 40, 20, 1, key="t5_L")
    nseed = cN[2].number_input("seed", 0, 999, 0, 1, key="t5_seed")

    fig5, ax5 = plt.subplots(figsize=(7.2, 3.3))
    summary = []
    for lbl, col in [("none", "#C0507A"), ("LayerNorm", "#1D9E75"), ("RMSNorm", "#185FA5")]:
        v = grad_norms(nact, nL, 64, nseed, lbl)
        ax5.semilogy(range(1, nL + 1), np.maximum(v, 1e-300), lw=2.1, marker="o", ms=3,
                     color=col, label=("no normalization" if lbl == "none" else lbl))
        summary.append((lbl, v[0], v[-1], v[-1] / v[0] if v[0] > 0 else float("inf")))
    ax5.axhline(1.0, color="0.7", lw=1, ls=":")
    ax5.set_xlabel("layer (1 = earliest)", fontsize=9)
    ax5.set_ylabel("‖gradient‖ (log)", fontsize=9)
    ax5.set_title(f"{nact} at depth {nL} — with and without normalization", fontsize=9.5)
    ax5.legend(fontsize=8); ax5.tick_params(labelsize=8)
    fig5.tight_layout(); st.pyplot(fig5, width="stretch")

    base = summary[0][1]

    def _vs(lbl, first):
        if lbl == "none":
            return "—"
        if base <= 0:
            return "**∞** (baseline is exactly 0)" if first > 0 else "both 0"
        return f"**{first / base:.1e}× more gradient**"

    st.markdown(
        "| setup | ‖grad‖ at layer 1 | shrink factor over the stack | vs. no-norm |\n|---|---|---|---|\n"
        + "\n".join(
            f"| **{('no normalization' if l == 'none' else l)}** | {a:.3e} | {r:.2e}× | "
            + _vs(l, a) + " |"
            for l, a, b_, r in summary))
    if base <= 0 and nact == "step":
        st.warning("**step** has derivative 0 everywhere, so *no* gradient reaches any layer — "
                   "with or without normalization. Normalization fixes *conditioning*; it cannot "
                   "conjure a gradient that does not exist. This is why the perceptron could not "
                   "be trained by backprop at all.", icon=":material/block:")
    st.info(
        "**The honest reading.** With **sigmoid at depth 20**, normalization recovers roughly "
        "**four orders of magnitude** of gradient — it does *not* make sigmoid as good as ReLU, "
        "but it turns *hopeless* into *trainable*. With **tanh** it flattens the curve almost "
        "completely. That is why the field's attention moved: a better activation buys you a few "
        "percent, while normalization buys you **the ability to train the network at all** — and "
        "it does so no matter which activation you picked.",
        icon=":material/insights:")
    st.markdown(
        "**The connection to Math.** Normalization is a **conditioning** fix. Badly scaled "
        "activations make the loss surface a **ravine** — a large ratio between the biggest and "
        "smallest curvature (the *condition number*, the eigenvalue spread of the Hessian, Math "
        "**X1 §13**). Gradient descent zig-zags down such a ravine (Math **X4** — try the "
        "**standardize features** toggle in its playground). Normalizing the activations "
        "re-rounds the bowl at *every layer*, which is the same medicine as feature scaling "
        "(ML **M7**), applied continuously throughout the network."
    )
