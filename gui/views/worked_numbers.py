"""Worked examples — every number, end to end (ANN module).

Four scenario-based walkthroughs with real arithmetic: one neuron, one layer (matrix
form), two layers (XOR), and a batch pass + one full training step. Inputs *and* every
weight/bias are adjustable, and each parameter is explained. Everything is computed live
with NumPy, so the displayed matrices and the step-by-step arithmetic can never drift apart.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

import lessons

# --------------------------------------------------------------------------- #
# LaTeX helpers — build matrices/vectors from real arrays (no hand-typed numbers)
# --------------------------------------------------------------------------- #
import mathviz as mv
from mathviz import bmat, rowvec, sigmoid, softmax  # noqa: F401  (used throughout)

_fmt = mv.fmt
_set_state = mv.set_state


st.title("Worked examples — every number, end to end")
st.caption("Four scenarios with the full arithmetic: one neuron → one layer (matrices) → "
           "two layers → a batch pass and one training step. Every input **and every "
           "weight/bias** is adjustable — each one is explained as you go.")

lessons.predict(
    "A layer with **3 neurons** and **2 inputs** — what shape is its weight matrix $W$, and how "
    "many numbers does the layer hold in total (including biases)?",
    "$W$ is **3×2** — *one row per neuron*, one column per input — plus a bias vector of length 3, "
    "so **6 + 3 = 9** learned numbers. The rule: a layer from $d$ inputs to $h$ neurons has a "
    "$h\\times d$ matrix and $h$ biases, i.e. $h(d+1)$ parameters.",
)

t1, t2, t3, t4 = st.tabs([
    "① One neuron", "② One layer (matrix)", "③ Two layers (XOR)", "④ Batch + a training step",
])

# =========================================================================== #
# ① ONE NEURON
# =========================================================================== #
with t1:
    st.markdown("#### 📋 Scenario — a loan-approval neuron")
    st.markdown(
        "A bank must answer one question about an applicant: **will they repay?** It has three "
        "facts, and one neuron to turn them into a probability. The **inputs** describe the "
        "applicant; the **weights and bias** *are the bank's policy* — and in a real network "
        "these are exactly the numbers **training would learn** from thousands of past loans."
    )
    st.markdown(
        "Each raw fact is first **scaled to 0–1** so the three are comparable (ML **M7**):\n\n"
        "| feature | raw quantity | scaled to | `0.00` means | `1.00` means |\n|---|---|---|---|---|\n"
        "| **income** | annual salary | salary ÷ 200 k, capped | no income | 200 k+ |\n"
        "| **credit** | credit score 300–850 | (score − 300) ÷ 550 | worst score | perfect score |\n"
        "| **debt** | debt-to-income ratio | ratio ÷ 1.0, capped | debt-free | debts = a full year's income |\n\n"
        f"So the defaults describe someone earning ~**{0.80*200:.0f} k**, with a **good** credit "
        f"score, whose debts are **30 %** of their income."
    )

    aug = st.toggle(
        "🎩 Show the bias as an **input** ($x_0=1$) — the *bias trick*", key="wn1_aug",
        help="The bias is mathematically just the weight on a constant input of 1. "
             "Turn this on to see it sitting in the input column like any other feature.")

    ctrl = st.columns([0.5, 0.5])
    with ctrl[0]:
        st.markdown("**👤 Inputs $\\mathbf x$** — this applicant")
        if aug:
            st.number_input("🔒 x₀ · constant input — locked at 1 *(this is the bias's input)*",
                            value=1.0, disabled=True, key="wn1_x0",
                            help="Deliberately not editable: the bias trick only works because "
                                 "this input is CONSTANT. Change the bias itself with the w₀ "
                                 "slider on the right.")
            st.caption("↑ **Why can't I move this?** Because it is not a feature of the "
                       "applicant — it is a *constant 1* that exists only so the bias has "
                       "something to multiply. Adjust **w₀** on the right instead: that is the "
                       "bias. *(Setting x₀ to some other constant c would change nothing real — "
                       "the network would just learn w₀/c instead.)*")
        x1 = st.slider("income (scaled)", 0.0, 1.0, 0.80, 0.05, key="wn1_x1")
        x2 = st.slider("credit score (scaled)", 0.0, 1.0, 0.60, 0.05, key="wn1_x2")
        x3 = st.slider("debt ratio (scaled)", 0.0, 1.0, 0.30, 0.05, key="wn1_x3")
    with ctrl[1]:
        # Seeded in session_state so the preset buttons can set them without Streamlit
        # warning about a widget having both a default and an API-set value.
        for _k, _v in {"wn1_w1": 2.0, "wn1_w2": 3.0, "wn1_w3": -4.0, "wn1_b": -1.0}.items():
            st.session_state.setdefault(_k, _v)
        st.markdown("**⚙️ Parameters $\\tilde{\\mathbf w}$** — the policy *(learned)*"
                    if aug else "**⚙️ Parameters $\\mathbf w, b$** — the policy *(learned)*")
        if aug:
            st.caption("↓ note this row lines up with the constant input on the left")
            b = st.slider("w₀ · weight on the constant input  ( = the bias )",
                          -8.0, 8.0, step=0.1, key="wn1_b")
        w1 = st.slider("w₁ · weight on income", -6.0, 6.0, step=0.1, key="wn1_w1")
        w2 = st.slider("w₂ · weight on credit", -6.0, 6.0, step=0.1, key="wn1_w2")
        w3 = st.slider("w₃ · weight on debt", -6.0, 6.0, step=0.1, key="wn1_w3")
        if not aug:
            b = st.slider("b · bias (how strict the bank is)", -8.0, 8.0, step=0.1, key="wn1_b")

    pc = st.columns(4)
    pc[0].button("↺ Defaults", key="wn1_p0", on_click=_set_state,
                 args=({"wn1_w1": 2.0, "wn1_w2": 3.0, "wn1_w3": -4.0, "wn1_b": -1.0},))
    pc[1].button("🔒 Strict bank", key="wn1_p1", on_click=_set_state,
                 args=({"wn1_w1": 2.0, "wn1_w2": 3.0, "wn1_w3": -4.0, "wn1_b": -4.0},))
    pc[2].button("🔓 Lenient bank", key="wn1_p2", on_click=_set_state,
                 args=({"wn1_w1": 2.0, "wn1_w2": 3.0, "wn1_w3": -4.0, "wn1_b": 2.0},))
    pc[3].button("💳 Credit-only", key="wn1_p3", on_click=_set_state,
                 args=({"wn1_w1": 0.0, "wn1_w2": 6.0, "wn1_w3": 0.0, "wn1_b": -3.0},))

    act_name, act = mv.activation_picker(
        "wn1_act", "sigmoid", "⚡ activation φ — how the score z becomes the output")

    x = np.array([x1, x2, x3])
    w = np.array([w1, w2, w3])
    contrib = w * x
    z = float(contrib.sum() + b)
    a = float(np.asarray(act["f"](np.array([z])), dtype=float)[0])
    names = ["income", "credit", "debt"]

    st.markdown("#### 🧠 What each parameter *means*")
    st.markdown(
        "| parameter | what it is | its sign says | its size says |\n|---|---|---|---|\n"
        f"| **w₁ = {w1:.1f}** (income) | the exchange rate from *income* to score | "
        f"{'higher income **helps**' if w1 > 0 else ('higher income **hurts**' if w1 < 0 else 'income is **ignored**')} | "
        f"one unit of income moves $z$ by **{w1:.1f}** |\n"
        f"| **w₂ = {w2:.1f}** (credit) | the exchange rate from *credit* to score | "
        f"{'higher credit **helps**' if w2 > 0 else ('higher credit **hurts**' if w2 < 0 else 'credit is **ignored**')} | "
        f"one unit of credit moves $z$ by **{w2:.1f}** |\n"
        f"| **w₃ = {w3:.1f}** (debt) | the exchange rate from *debt* to score | "
        f"{'higher debt **helps**' if w3 > 0 else ('higher debt **hurts**' if w3 < 0 else 'debt is **ignored**')} | "
        f"one unit of debt moves $z$ by **{w3:.1f}** |\n"
        f"| **b = {b:.1f}** (bias) | the score for an all-zero applicant | "
        f"{'the bank starts **sceptical**' if b < 0 else ('the bank starts **generous**' if b > 0 else 'the bank starts **neutral**')} | "
        f"shifts *every* decision by **{b:.1f}** |"
    )
    with st.expander("📖 Deeper: weights, bias, and why these four numbers are all there is"):
        st.markdown(
            "**A weight is a rate of exchange.** Because $z=\\sum_i w_i x_i + b$, the partial "
            "derivative $\\partial z/\\partial x_i = w_i$: *if that feature goes up by one unit, "
            "the score moves by exactly $w_i$*. So a weight has two independent jobs — its "
            "**sign** says which way the evidence points (positive = evidence *for* approval, "
            "negative = evidence *against*), and its **magnitude** says how loudly that feature "
            "speaks. `w₃ = −4.0` means debt is both the *strongest* signal here and a "
            "*disqualifying* one.\n\n"
            "**Magnitudes are only comparable when the features are.** We scaled all three "
            "inputs to 0–1 first. Without that, a weight of 2.0 on *income in dollars* and 3.0 on "
            "*credit score in points* would be meaningless to compare — which is exactly why "
            "**feature scaling** matters (ML **M7**).\n\n"
            "**The bias is the neuron's default opinion.** Set every feature to zero and "
            f"$z=b={b:.1f}$ — that is the score of a blank applicant. Equivalently it moves the "
            "**decision boundary**: the neuron flips its answer where $z=0$, i.e. on the plane "
            "$\\mathbf w\\cdot\\mathbf x + b = 0$. Raising $b$ slides that plane so *more* "
            "applicants land on the approve side (try **Lenient bank**), lowering it approves "
            "fewer (**Strict bank**). The weights **tilt** the boundary; the bias **shifts** it — "
            "the same split you met in Math **X1**.\n\n"
            "**Why exactly four numbers?** A neuron with $d$ inputs has $d$ weights $+\\,1$ bias "
            f"$= d+1$ parameters (here $3+1=4$). Nothing else about it is adjustable — the "
            "activation is a fixed function. So 'learning' *is* choosing these four numbers.\n\n"
            f"**And this is what backprop computes.** The gradient of the score with respect to a "
            f"weight is $\\partial z/\\partial w_i = x_i$ — the input itself. So a feature that is "
            f"large for this applicant (here income $={x1:.2f}$) is a weight the training step "
            "will move the most. That is the whole reason inputs show up in the weight-gradient "
            "formula on the **Backprop** page."
        )

    with st.expander("📏 How big should a weight be? Does **−4 … 4** mean anything physical?"):
        st.markdown(
            "**Yes — with a sigmoid output, a weight has exact units: log-odds per unit of "
            "feature.** The neuron's $z$ is not a probability, it is the **log-odds** (the "
            "*logit*) of the outcome:"
        )
        st.latex(r"a=\sigma(z)\iff z=\ln\frac{a}{1-a},\qquad \text{odds}=\frac{a}{1-a}=e^{z}")
        st.markdown(
            "So $w_i$ answers: *if this feature rises by one unit, by how much does the "
            "**log-odds** move?* Exponentiate it and you get the **odds ratio** — the factor by "
            "which the odds are multiplied. That is the physical meaning, and it is exactly how "
            "coefficients are read in **logistic regression** (ML **M2**):"
        )
        st.markdown(
            "| weight $w$ | $e^{w}$ = odds ratio | reading (for a full 0→1 swing in that feature) |\n"
            "|---|---|---|\n"
            "| 0 | 1.0× | the feature is **ignored** |\n"
            "| 0.5 | 1.6× | a nudge |\n"
            "| 1 | 2.7× | meaningful |\n"
            "| 2 | 7.4× | strong |\n"
            "| 4 | 55× | dominant — hard for others to outvote |\n"
            "| −4 | 0.018× (1/55) | dominant, and **against** |\n\n"
            f"**Your current weights read as:** income ×**{np.exp(w1):.1f}**, credit "
            f"×**{np.exp(w2):.1f}**, debt ×**{np.exp(w3):.3f}** on the odds of repaying, for a "
            "full 0→1 move in each feature."
        )
        st.markdown(
            "**Why the sliders stop near ±4–6 — saturation.** Since every input is in $[0,1]$, "
            "$z$ can only reach about $\\sum_i|w_i|+|b|$. And the sigmoid *runs out of room* "
            "fast: $\\sigma(5)=0.993$, $\\sigma(10)=0.99995$. Past $|z|\\approx6$ the output is "
            "pinned at 0 or 1, extra weight changes **nothing visible** — and worse, the "
            "gradient $\\sigma'(z)=\\sigma(1-\\sigma)$ collapses toward 0, so the neuron stops "
            "learning (the vanishing-gradient effect, neuron lesson §6). So ±4 per feature spans "
            "the whole useful story: **0 = ignored → ±4 = decisive**. Larger values are not "
            "wrong, just pointless here."
        )
        st.markdown(
            "**Is there a 'correct' magnitude? Only relative to the feature's scale.** A weight "
            "is meaningless on its own — only the product $w_i x_i$ is. Measure income in "
            "*dollars* instead of a 0–1 score and the same behaviour needs a weight around "
            "$10^{-5}$. Rescaling the input by $c$ just rescales the learned weight by $1/c$. "
            "That is precisely why we **standardize features** before training (**M7**) — it "
            "puts all weights on a comparable footing so their magnitudes can be compared at all."
        )
        st.markdown(
            "**In practice nobody picks these numbers by hand.** The real pipeline is:\n\n"
            "1. **Initialize** small and random — e.g. Xavier/He scaling, roughly "
            "$\\sigma\\sim\\sqrt{1/n_{\\text{in}}}$ (experiment **e10**). Never all-zero (every "
            "neuron would stay identical), never huge (instant saturation).\n"
            "2. **Learn** by gradient descent: $w_i \\leftarrow w_i - \\eta\\,\\partial L/\\partial w_i$, "
            "repeated over the data (tab ④ does one such step by hand).\n"
            "3. **Restrain** with **weight decay / L2**, which actively pulls weights toward 0 — "
            "big weights mean a sharp, high-variance boundary that overfits (**Regularization**).\n\n"
            "The upshot: in a *trained* deep net, typical weights are **small** — often "
            "$|w|\\approx0.01$–$1$ — and the work is done by *many* of them together. Big, "
            "human-legible weights like $-4$ show up mainly in tiny hand-built nets like this "
            "one and in the logic-gate constructions on the *Neurons → computer* page."
        )

    st.markdown("#### 🧮 Step 1 — the weighted sum (a dot product)")
    if aug:
        st.caption("With the bias as an input there is **no `+ b` term** — it is inside the dot product.")
        st.latex(r"z = \tilde{\mathbf w}\cdot\tilde{\mathbf x} = " +
                 rowvec(np.append(b, w), 1) + r"\cdot" + bmat(np.append(1.0, x)))
        st.latex(f"z = ({b:.1f})(1.00) + ({w1:.1f})({x1:.2f}) + ({w2:.1f})({x2:.2f}) + ({w3:.1f})({x3:.2f})"
                 r"\;=\;" + f"{b:.2f} + {contrib[0]:.2f} + {contrib[1]:.2f} + {contrib[2]:.2f} = {z:.3f}")
    else:
        st.latex(r"z = \mathbf w\cdot\mathbf x + b = " + rowvec(w, 1) + r"\cdot" + bmat(x) + " + (" + _fmt(b, 1) + ")")
    if not aug:
        st.latex(
            r"z = " +
            f"({w1:.1f})({x1:.2f}) + ({w2:.1f})({x2:.2f}) + ({w3:.1f})({x3:.2f}) + ({b:.1f})"
            r"\;=\;" + f"{contrib[0]:.2f} + {contrib[1]:.2f} + {contrib[2]:.2f} + ({b:.2f})"
            r"\;=\;" + f"{z:.3f}"
        )

    st.markdown("**Who actually drove this decision?** Each bar is one term $w_i x_i$:")
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    labels = ([f"{n}\n({w[i]:.1f} × {x[i]:.2f})" for i, n in enumerate(names)]
              + [(f"x₀ = 1\n({b:.1f} × 1.00)" if aug else f"bias\n({b:.1f})")])
    vals = list(contrib) + [b]
    cols = ["#1D9E75" if v >= 0 else "#C0507A" for v in vals]
    ax.barh(labels, vals, color=cols, height=0.6)
    ax.axvline(0, color="0.35", lw=1.1)
    # Leave room on both sides so the value labels sit *outside* the bars, never clipped.
    span = max(max(abs(v) for v in vals), 0.5)
    pad = span * 0.05
    ax.set_xlim(-span * 1.5, span * 1.5)
    for i, v in enumerate(vals):
        ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.2f}", va="center", fontsize=9,
                ha="left" if v >= 0 else "right", color="#33312E")
    ax.set_xlabel("contribution to z", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.margins(y=0.14)
    ax.set_title(f"total z = {z:.3f}", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    with st.expander("🎩 Can the bias just be **another input**? (the *bias trick*)"):
        xt = np.append(x, 1.0)
        wt = np.append(w, b)
        st.markdown(
            "**Yes — and it is a standard move.** Bolt a constant **1** onto the input vector and "
            "put the bias into the weight vector. The bias is then simply *the weight on an input "
            "that is always 1*:"
        )
        st.latex(r"\tilde{\mathbf x} = \begin{bmatrix}\mathbf x \\ 1\end{bmatrix} = " + bmat(xt) +
                 r",\qquad \tilde{\mathbf w} = \begin{bmatrix}\mathbf w \\ b\end{bmatrix} = " + bmat(wt, 1))
        st.latex(r"\tilde{\mathbf w}\cdot\tilde{\mathbf x} = " +
                 " + ".join(f"({wt[i]:.1f})({xt[i]:.2f})" for i in range(4)) +
                 f" = {float(wt @ xt):.3f}" +
                 r"\;=\; \mathbf w\cdot\mathbf x + b \;=\; " + f"{z:.3f}" + r"\quad\checkmark")
        st.markdown(
            "Identical number, one cleaner formula — no more `+ b` special case. **Why it is "
            "worth knowing:**\n\n"
            "- **The maths gets uniform.** Every layer is *just* a dot product (or matrix "
            "product), which is why textbook derivations of the perceptron rule and of least "
            "squares use it.\n"
            f"- **It explains the bias gradient.** For any weight, $\\partial z/\\partial w_i = "
            f"x_i$. The bias's 'input' is the constant 1, so $\\partial z/\\partial b = 1$ — "
            "exactly the rule you use in backprop, now with no special case to memorise.\n"
            "- **You have already seen it.** In ML **M1**, linear regression prepends a **column "
            "of ones** to the design matrix $X$ so that $\\boldsymbol\\beta$ contains the "
            "intercept. Same trick. It is also neuron-lesson **Task 7**.\n"
            "- **It is why the bias is a bar in the chart above** — it contributes to $z$ exactly "
            "like a feature term, it just never varies between applicants.\n\n"
            "**So why do PyTorch/Keras still keep `bias` separate?** Three practical reasons:\n\n"
            "1. **Batching.** With $N$ samples you would have to glue a ones-column onto every "
            "input matrix; adding a bias *vector* by **broadcasting** is cheaper and simpler.\n"
            "2. **Regularization must treat it differently.** Weight decay shrinks weights toward "
            "0 — but shrinking the *bias* would drag the decision boundary toward the origin for "
            "no statistical reason. Frameworks routinely exclude biases from weight decay, which "
            "is only possible if they are separate objects.\n"
            "3. **Initialization differs.** Weights start random (Xavier/He); biases usually "
            "start at exactly **0**.\n\n"
            "*Mathematically identical, operationally distinct* — a good example of theory and "
            "implementation diverging for practical reasons."
        )

    st.markdown("#### 🧮 Step 2 — the activation: turning the score into an output")
    st.latex(r"\varphi(z) = " + act["tex"])
    st.latex(r"a = \varphi(" + f"{z:.3f}" + ") = " + f"{a:.4f}")
    mv.activation_explainer(act_name, z_mark=z)

    approve = z > 0
    m = st.columns(3)
    m[0].metric("z (pre-activation)", f"{z:.3f}")
    m[1].metric(f"a = {act_name}(z)", f"{a:.4f}")
    m[2].metric("decision (z > 0)", "APPROVE" if approve else "DECLINE")
    (st.success if approve else st.warning)(
        (f"The neuron outputs **{a:.4f}** → " +
         (f"**approve** — with a sigmoid this reads as a **{a*100:.0f}% chance** of repaying."
          if act_name == "sigmoid" else
          f"**decline** — with a sigmoid this reads as only a **{a*100:.0f}% chance** of repaying."))
        if act_name == "sigmoid" else
        (f"The neuron outputs **{a:.4f}** → **{'approve' if approve else 'decline'}**. "
         f"Note: with **{act_name}** the output is *not* a probability — the decision still flips "
         f"where $z=0$, because every activation here is monotonic, but only **sigmoid** turns the "
         f"score into a calibrated 0–1 chance."),
        icon=":material/account_balance:",
    )
    st.markdown("#### 🔍 Reading the result — and what would flip it")
    if act_name != "sigmoid":
        st.caption(f"*(The odds reading below assumes a **sigmoid** output; you currently have "
                   f"**{act_name}**. The break-even analysis holds for any monotonic activation, "
                   f"since the decision always flips at $z=0$.)*")
    nw = float(np.linalg.norm(w))
    rc = st.columns(3)
    rc[0].metric("log-odds z", f"{z:+.3f}")
    rc[1].metric("odds of repaying", f"{np.exp(z):.2f} : 1")
    rc[2].metric("distance to boundary", f"{z/nw:+.3f}" if nw > 1e-9 else "—")
    st.markdown(
        f"The decision flips exactly at $z=0$ (where $\\sigma(0)=0.5$). This applicant sits "
        f"**{abs(z):.3f} log-odds {'above' if z >= 0 else 'below'}** that line, i.e. odds of about "
        f"**{np.exp(z):.2f} to 1** {'in favour of' if z >= 0 else 'against'} repaying. The "
        f"*geometric* distance to the boundary plane is $z/\\lVert\\mathbf w\\rVert = "
        f"{z/nw:+.3f}$ (Math **X1** §11) — small numbers mean a borderline case the model is "
        f"unsure about."
    )
    rows = []
    for i, n in enumerate(names):
        rest = z - contrib[i]  # everything except this feature (bias included)
        if abs(w[i]) < 1e-9:
            rows.append(f"| **{n}** | {x[i]:.2f} | — | weight is 0, so this feature can *never* change the outcome |")
        else:
            xs = -rest / w[i]
            ok = 0.0 <= xs <= 1.0
            rows.append(
                f"| **{n}** | {x[i]:.2f} | **{xs:+.2f}** | "
                + ("✅ reachable — moving this feature alone flips the decision"
                   if ok else "❌ outside 0–1, so this feature alone *cannot* flip it") + " |"
            )
    st.markdown(
        "**Break-even:** holding the other two fixed, what would this feature have to be for "
        "$z$ to hit 0?\n\n"
        "| feature | now | flips at | |\n|---|---|---|---|\n" + "\n".join(rows)
    )
    st.caption("Each break-even solves $w_i x_i^\\star + (\\text{everything else}) = 0$, i.e. "
               "$x_i^\\star = -(\\text{rest})/w_i$ — the point where this one feature is just "
               "enough to tip the balance.")

    st.info("**Try it:** set every weight to 0 — the output collapses to $\\sigma(b)$, the bank's "
            "prejudice with no evidence. Or flip `w₃` positive and watch debt start *helping*: "
            "the neuron has no idea what 'debt' means, it only knows the number you gave it. "
            "*Meaning lives entirely in the weights.*", icon=":material/lightbulb:")

# =========================================================================== #
# ② ONE LAYER — MATRIX FORM
# =========================================================================== #
with t2:
    st.markdown("#### 📋 Scenario — classify a flower into 3 species")
    st.markdown(
        "Now we want **3 outputs** (one score per species) from **2 measurements**. That means "
        "**3 neurons side by side** — and stacking their weight vectors as *rows* gives a "
        "**matrix** $W$. Each row below is one neuron: its own two weights and its own bias."
    )

    cx = st.columns(2)
    p_len = cx[0].slider("petal length (cm)", 1.0, 7.0, 4.5, 0.1, key="wn2_len")
    p_wid = cx[1].slider("petal width (cm)", 0.1, 2.6, 1.5, 0.1, key="wn2_wid")

    names = ["setosa", "versicolor", "virginica"]
    st.markdown("**⚙️ The layer's 9 parameters — one row (2 weights + 1 bias) per neuron**")
    cols = st.columns(3)
    Wl, bl = [], []
    defaults = [(-1.0, -1.5, 4.0), (0.6, -0.4, -1.0), (0.9, 1.6, -4.5)]
    for i, (nm, (dw1, dw2, db)) in enumerate(zip(names, defaults)):
        st.session_state.setdefault(f"wn2_w{i}1", dw1)
        st.session_state.setdefault(f"wn2_w{i}2", dw2)
        st.session_state.setdefault(f"wn2_b{i}", db)
        with cols[i]:
            st.markdown(f"**Neuron {i+1} — {nm}**")
            a1 = st.slider(f"w[{i+1},1] · length", -3.0, 3.0, step=0.1, key=f"wn2_w{i}1")
            a2 = st.slider(f"w[{i+1},2] · width", -3.0, 3.0, step=0.1, key=f"wn2_w{i}2")
            bb = st.slider(f"b[{i+1}] · bias", -8.0, 8.0, step=0.1, key=f"wn2_b{i}")
            Wl.append([a1, a2])
            bl.append(bb)
    st.button("↺ Reset all 9 to defaults", key="wn2_reset", on_click=_set_state, args=({
        f"wn2_w{i}{j+1}": defaults[i][j] for i in range(3) for j in range(2)
    } | {f"wn2_b{i}": defaults[i][2] for i in range(3)},))

    x = np.array([p_len, p_wid])
    W = np.array(Wl)
    b = np.array(bl)
    z = W @ x + b
    p = softmax(z)

    st.markdown("#### 🧠 What each parameter *means* here")
    st.markdown(
        "| | meaning | in this layer |\n|---|---|---|\n"
        "| **a row of $W$** | one neuron = one class's own linear scorer | row 1 scores *setosa*, "
        "row 2 *versicolor*, row 3 *virginica* |\n"
        "| **a column of $W$** | how one *feature* feeds every class | column 1 = petal length, "
        "column 2 = petal width |\n"
        f"| **$w_{{ij}}$** | how much feature $j$ raises class $i$'s score | e.g. $w_{{32}}="
        f"{W[2,1]:.1f}$: width "
        f"{'boosts' if W[2,1] > 0 else 'lowers'} *virginica* |\n"
        f"| **$b_i$** | class $i$'s **base score** (a prior / how common it is) | "
        f"setosa {b[0]:+.1f}, versicolor {b[1]:+.1f}, virginica {b[2]:+.1f} |"
    )
    with st.expander("📖 Deeper: rows, biases, softmax, and where the boundaries come from"):
        st.markdown(
            "**Three neurons, three independent scorers.** Nothing couples the rows during the "
            "linear step — neuron 2 never sees neuron 1's weights. Row $i$ just computes its own "
            "dot product $\\mathbf w_i\\cdot\\mathbf x + b_i$. Writing them as a matrix is pure "
            "bookkeeping: it lets one $W\\mathbf x$ do all three at once (and lets a GPU do "
            "thousands).\n\n"
            "**A bias is a class prior.** With both measurements at zero, the scores are exactly "
            f"$\\mathbf b = ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f})$. A class with a big bias wins "
            "*by default* and needs less evidence — which is how a model encodes 'setosa is "
            "common' or 'this class is rare'. Drag one bias up and watch that species take over.\n\n"
            "**Only score *differences* matter.** Softmax is shift-invariant: add the same "
            "constant to all three scores and the probabilities are unchanged "
            "($e^{z_i+c}/\\sum_j e^{z_j+c}=e^{z_i}/\\sum_j e^{z_j}$). So the layer has one "
            "redundant degree of freedom — and this is also *why* the stable implementation may "
            "subtract $\\max z$ for free (Math **X6**).\n\n"
            "**Where do the decision boundaries come from?** Class $i$ beats class $j$ exactly "
            "when $z_i>z_j$, i.e. $(\\mathbf w_i-\\mathbf w_j)\\cdot\\mathbf x + (b_i-b_j) > 0$ — "
            "a **straight line** in the feature plane. Three classes therefore carve the plane "
            "into (at most) three regions with straight edges, shown in the map below. Changing a "
            "*weight* rotates a boundary; changing a *bias* slides it — same as the single neuron, "
            "now once per pair of classes.\n\n"
            f"**Parameter count.** $3\\times2$ weights $+\\,3$ biases $=$ **9 numbers**, the "
            "general rule being $h(d+1)$ for $d$ inputs and $h$ neurons. Every one of them would "
            "be learned by gradient descent from labelled flowers."
        )

    st.markdown("#### 🔢 The weight matrix — one row per neuron")
    st.latex(
        r"W = " + bmat(W, 1) + r"\quad(3\times 2),\qquad \mathbf b = " + bmat(b, 1) +
        r"\quad(3),\qquad \mathbf x = " + bmat(x) + r"\quad(2)"
    )

    st.markdown("#### 🧮 Step 1 — one matrix-vector product = three dot products")
    st.latex(r"\mathbf z = W\mathbf x + \mathbf b = " + bmat(W, 1) + bmat(x) + " + " + bmat(b, 1))
    for i in range(3):
        st.latex(
            f"z_{i+1} = ({W[i,0]:.1f})({x[0]:.2f}) + ({W[i,1]:.1f})({x[1]:.2f}) + ({b[i]:.1f})"
            r"\;=\;" + f"{z[i]:.3f}" + r"\quad\text{(" + names[i] + ")}"
        )
    st.latex(r"\mathbf z = " + bmat(z, 3))

    st.markdown("#### 🧮 Step 2 — softmax turns scores into probabilities")
    st.latex(r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}")
    ex = np.exp(z - z.max())
    st.latex(r"e^{z_i - \max z} = " + rowvec(ex, 4) + r",\qquad \textstyle\sum_j = " + f"{ex.sum():.4f}")
    st.latex(r"\mathbf p = " + rowvec(p, 4) + r",\qquad \textstyle\sum_i p_i = " + f"{p.sum():.2f}")

    m = st.columns(3)
    for i in range(3):
        m[i].metric(names[i], f"{p[i]*100:.1f}%")
    st.success(f"Prediction: **{names[int(p.argmax())]}** "
               f"({p.max()*100:.1f}% confident) — the largest score wins.",
               icon=":material/local_florist:")

    st.markdown("#### 🗺️ The decision map these 9 numbers create")
    gl = np.linspace(1.0, 7.0, 220)
    gw = np.linspace(0.1, 2.6, 220)
    GL, GW = np.meshgrid(gl, gw)
    grid = np.c_[GL.ravel(), GW.ravel()]
    winner = (grid @ W.T + b).argmax(axis=1).reshape(GL.shape)
    fig2, ax2 = plt.subplots(figsize=(5.6, 3.4))
    ax2.contourf(GL, GW, winner, levels=[-0.5, 0.5, 1.5, 2.5],
                 colors=["#FBEAF0", "#E1F5EE", "#E6F1FB"])
    ax2.contour(GL, GW, winner, levels=[0.5, 1.5], colors="#6B6A66", linewidths=1.2)
    # Label only regions big enough to hold text, so labels never pile up on each other.
    for i, nm in enumerate(names):
        mask = winner == i
        if mask.mean() > 0.05:
            ax2.text(GL[mask].mean(), GW[mask].mean(), nm, fontsize=9, ha="center", va="center",
                     color="#33312E",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65))
    ax2.plot(p_len, p_wid, "o", ms=11, color="#33312E", zorder=5)
    # Keep the callout inside the axes: flip its side near the right/top edge.
    dx, ha = (-12, "right") if p_len > 5.5 else (12, "left")
    dy = -16 if p_wid > 2.2 else 12
    ax2.annotate(f"this flower → {names[int(p.argmax())]}", (p_len, p_wid),
                 textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9C9B95", alpha=0.9))
    ax2.set_xlabel("petal length (cm)")
    ax2.set_ylabel("petal width (cm)")
    ax2.set_title("who wins where — boundaries are straight lines", fontsize=9)
    st.pyplot(fig2, width="stretch")
    st.info("**Move a weight and a boundary rotates; move a bias and it slides.** That is the "
            "entire geometry of a linear layer: 9 numbers → 3 straight-edged territories. To get "
            "*curved* boundaries you need a hidden layer — which is tab ③.",
            icon=":material/lightbulb:")

# =========================================================================== #
# ③ TWO LAYERS — XOR
# =========================================================================== #
with t3:
    st.markdown("#### 📋 Scenario — XOR, the problem one layer cannot solve")
    st.markdown(
        "XOR is 1 when the inputs **differ**. No single straight line separates it (see *Two "
        "neurons*), so we need a **hidden layer** to build new features first. Here is a "
        "hand-built 2-layer net that solves it exactly — with **ReLU**, $\\varphi(v)=\\max(0,v)$."
    )
    c = st.columns(2)
    i1 = c[0].segmented_control("x₁", [0, 1], default=1, key="wn3_x1")
    i2 = c[1].segmented_control("x₂", [0, 1], default=0, key="wn3_x2")
    x = np.array([float(i1 if i1 is not None else 1), float(i2 if i2 is not None else 0)])

    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([1.0, -2.0])
    b2 = 0.0

    z1 = W1 @ x + b1
    h = np.maximum(0.0, z1)
    y = float(W2 @ h + b2)

    st.markdown("#### 🔢 The two layers")
    st.latex(r"W^{(1)} = " + bmat(W1, 1) + r",\; \mathbf b^{(1)} = " + bmat(b1, 1) +
             r"\qquad W^{(2)} = " + rowvec(W2, 1) + r",\; b^{(2)} = " + _fmt(b2, 1))
    st.caption("Hidden unit h₁ = ReLU(x₁+x₂) fires when *at least one* input is on (an OR-ish "
               "unit). Hidden unit h₂ = ReLU(x₁+x₂−1) fires only when *both* are on (an AND-ish "
               "unit). The output subtracts twice the AND — vetoing the (1,1) corner.")

    st.markdown(f"#### 🧮 Forward pass for $\\mathbf x = ({x[0]:.0f}, {x[1]:.0f})$")
    st.markdown("**Layer 1 — hidden**")
    st.latex(r"\mathbf z^{(1)} = W^{(1)}\mathbf x + \mathbf b^{(1)} = " + bmat(W1, 1) + bmat(x, 0) +
             " + " + bmat(b1, 1) + " = " + bmat(z1, 1))
    for i in range(2):
        st.latex(f"z^{{(1)}}_{i+1} = (1)({x[0]:.0f}) + (1)({x[1]:.0f}) + ({b1[i]:.0f}) = {z1[i]:.0f}"
                 r"\;\longrightarrow\; h_" + str(i + 1) +
                 f" = \\max(0, {z1[i]:.0f}) = {h[i]:.0f}")
    st.latex(r"\mathbf h = " + bmat(h, 0))
    st.markdown("**Layer 2 — output**")
    st.latex(r"y = W^{(2)}\mathbf h + b^{(2)} = " + rowvec(W2, 1) + bmat(h, 0) + " + " +
             _fmt(b2, 1) + " = " + f"({W2[0]:.0f})({h[0]:.0f}) + ({W2[1]:.0f})({h[1]:.0f})" +
             r" = " + f"{y:.0f}")

    m = st.columns(4)
    m[0].metric("h₁ (OR-ish)", f"{h[0]:.0f}")
    m[1].metric("h₂ (AND-ish)", f"{h[1]:.0f}")
    m[2].metric("output y", f"{y:.0f}")
    m[3].metric("XOR target", f"{int(x[0] != x[1])}")

    rows = []
    for a_, b_ in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        xx = np.array([a_, b_], float)
        zz = W1 @ xx + b1
        hh = np.maximum(0, zz)
        yy = W2 @ hh + b2
        rows.append(f"| {a_} | {b_} | {zz[0]:.0f} | {zz[1]:.0f} | {hh[0]:.0f} | {hh[1]:.0f} | "
                    f"**{yy:.0f}** | {int(a_ != b_)} |")
    st.markdown("#### ✅ All four cases")
    st.markdown("| x₁ | x₂ | z₁ | z₂ | h₁ | h₂ | y | XOR |\n|---|---|---|---|---|---|---|---|\n"
                + "\n".join(rows))
    st.success("**4/4 correct.** The magic is h₂: it only fires at (1,1), and the output weight "
               "**−2** uses it to cancel the OR — turning OR into XOR. The hidden layer *invented "
               "a feature* that made the problem linearly separable.", icon=":material/check_circle:")

# =========================================================================== #
# ④ BATCH + ONE TRAINING STEP
# =========================================================================== #
with t4:
    st.markdown("#### 📋 Part A — a whole batch in one matrix multiply")
    st.markdown(
        "Real training pushes **many samples at once**. Stack them as rows of $X$ "
        "($N$ samples × $d$ features) and the layer becomes $Z = XW^\\top + \\mathbf b$ — "
        "the bias is **broadcast** to every row."
    )
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([1.0, -2.0])
    Z1 = X @ W1.T + b1
    H = np.maximum(0.0, Z1)
    Y = H @ W2

    st.latex(r"\underbrace{" + bmat(X, 0) + r"}_{X\,(4\times 2)}\underbrace{" + bmat(W1.T, 0) +
             r"}_{W^{(1)\top}(2\times 2)} + \underbrace{" + rowvec(b1, 0) +
             r"}_{\text{broadcast}} = \underbrace{" + bmat(Z1, 0) + r"}_{Z^{(1)}(4\times 2)}")
    st.latex(r"H = \mathrm{ReLU}(Z^{(1)}) = " + bmat(H, 0) +
             r",\qquad \mathbf y = H\,W^{(2)\top} = " + bmat(Y, 0))
    st.info("All four XOR samples solved in **two matrix multiplies**. This is why GPUs matter: "
            "the batch dimension turns a loop into one big matmul. Shape rule: "
            "$(4\\times2)(2\\times2)\\to(4\\times2)$, then $(4\\times2)(2\\times1)\\to(4\\times1)$.",
            icon=":material/memory:")

    st.divider()
    st.markdown("#### 📋 Part B — one full training step (forward → loss → backward → update)")
    st.markdown(
        "Now the weights are **not** hand-built — we *learn* them. A 2-layer net with **tanh** "
        "hidden units and a **sigmoid** output, trained on a single example with squared error."
    )
    lr = st.slider("learning rate η", 0.1, 2.0, 0.5, 0.1, key="wn4_lr")

    x = np.array([1.0, 0.0])
    y_t = 1.0
    W1t = np.array([[0.5, -0.2], [0.3, 0.8]])
    b1t = np.array([0.1, -0.1])
    W2t = np.array([0.7, -0.4])
    b2t = 0.2

    z1 = W1t @ x + b1t
    h = np.tanh(z1)
    z2 = float(W2t @ h + b2t)
    a = float(sigmoid(z2))
    L = (a - y_t) ** 2

    st.markdown("**1 · Forward**")
    st.latex(r"\mathbf z^{(1)} = " + bmat(W1t) + bmat(x, 0) + " + " + bmat(b1t) + " = " + bmat(z1))
    st.latex(r"\mathbf h = \tanh(\mathbf z^{(1)}) = " + bmat(h, 4))
    st.latex(r"z^{(2)} = " + rowvec(W2t) + bmat(h, 4) + " + " + _fmt(b2t) + " = " + f"{z2:.4f}"
             r"\qquad a = \sigma(z^{(2)}) = " + f"{a:.4f}")
    st.latex(r"L = (a - y)^2 = (" + f"{a:.4f} - {y_t:.0f})^2 = {L:.4f}")

    dLda = 2 * (a - y_t)
    dadz2 = a * (1 - a)
    d2 = dLda * dadz2
    gW2 = d2 * h
    gb2 = d2
    dh = d2 * W2t
    dz1 = dh * (1 - h ** 2)
    gW1 = np.outer(dz1, x)
    gb1 = dz1

    st.markdown("**2 · Backward** — the chain rule, layer by layer")
    st.latex(r"\frac{\partial L}{\partial a} = 2(a-y) = " + f"{dLda:.4f}"
             r",\qquad \frac{\partial a}{\partial z^{(2)}} = a(1-a) = " + f"{dadz2:.4f}")
    st.latex(r"\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = " + f"{dLda:.4f}\\times{dadz2:.4f} = {d2:.4f}")
    st.latex(r"\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)}\mathbf h^\top = " + rowvec(gW2, 4) +
             r",\qquad \frac{\partial L}{\partial b^{(2)}} = " + f"{gb2:.4f}")
    st.latex(r"\boldsymbol\delta^{(1)} = (\delta^{(2)}W^{(2)})\odot(1-\mathbf h^2) = " + bmat(dz1, 4))
    st.latex(r"\frac{\partial L}{\partial W^{(1)}} = \boldsymbol\delta^{(1)}\mathbf x^\top = " + bmat(gW1, 4))

    W1n = W1t - lr * gW1
    b1n = b1t - lr * gb1
    W2n = W2t - lr * gW2
    b2n = b2t - lr * gb2
    h_new = np.tanh(W1n @ x + b1n)
    a_new = float(sigmoid(W2n @ h_new + b2n))
    L_new = (a_new - y_t) ** 2

    st.markdown("**3 · Update** — one gradient-descent step, $\\theta \\leftarrow \\theta - \\eta\\,\\partial L/\\partial\\theta$")
    st.latex(r"W^{(2)}_{\text{new}} = " + rowvec(W2t) + r" - " + f"{lr:.1f}" + rowvec(gW2, 4) +
             " = " + rowvec(W2n, 4))
    st.latex(r"W^{(1)}_{\text{new}} = " + bmat(W1t) + r" - " + f"{lr:.1f}" + bmat(gW1, 4) +
             " = " + bmat(W1n, 4))

    m = st.columns(3)
    m[0].metric("output before", f"{a:.4f}")
    m[1].metric("output after", f"{a_new:.4f}", delta=f"{a_new - a:+.4f}")
    m[2].metric("loss", f"{L_new:.4f}", delta=f"{L_new - L:+.4f}", delta_color="inverse")
    (st.success if L_new < L else st.warning)(
        f"Loss went **{L:.4f} → {L_new:.4f}** in one step (target = {y_t:.0f}). "
        + ("That is learning: repeat this a few thousand times and the net fits the data."
           if L_new < L else
           "Too large a learning rate **overshoots** — lower η and the loss falls again."),
        icon=":material/trending_down:",
    )
    st.info("Every gradient above is what **`core.engine.Value.backward()`** computes for you "
            "(see the **Backprop** page) — here it is by hand so you can see there is no magic: "
            "just the chain rule multiplying local derivatives back through the layers.",
            icon=":material/lightbulb:")
