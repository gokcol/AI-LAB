"""Worked examples II — bigger networks, still every number (ANN module).

Four escalating scenarios, all computed live with NumPy:
  ① 2 inputs → 1 output   (umbrella; the decision line is drawable)
  ② 4 inputs → 1 output   (spam filter; contribution breakdown)
  ③ 5 inputs → 2 outputs  (triage; a 2×5 matrix, softmax)
  ④ 5 → 3 → 2             (same triage with a hidden layer that builds syndromes)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import mathviz as mv

st.title("Worked examples II — bigger networks")
st.caption("The same by-hand arithmetic as the first page, scaled up: 2→1, 4→1, 5→2, and "
           "5→3→2. Watch what changes as you add inputs, add outputs, then add a layer.")

lessons.predict(
    "Going from **5 inputs → 2 outputs** in one layer, to **5 → 3 → 2** with a hidden layer: "
    "how many learned numbers does each hold?",
    "One layer: $2\\times5 + 2 = \\mathbf{12}$. Two layers: $(3\\times5+3) + (2\\times3+2) = "
    "18+8 = \\mathbf{26}$. The hidden layer more than doubles the parameters — and buys the "
    "ability to combine features **non-linearly**, which is the only reason to pay for it.",
)

t1, t2, t3, t4 = st.tabs([
    "① 2 → 1", "② 4 → 1", "③ 5 → 2 (one layer)", "④ 5 → 3 → 2 (two layers)",
])

# =========================================================================== #
# ① 2 INPUTS → 1 OUTPUT
# =========================================================================== #
with t1:
    st.markdown("#### 📋 Scenario — should I take an umbrella?")
    st.markdown(
        "Two inputs is the sweet spot for *seeing* what a neuron does: with two features the "
        "decision boundary is a **line you can draw**. Both inputs are scaled 0–1.\n\n"
        "| feature | `0.00` | `1.00` |\n|---|---|---|\n"
        "| **cloud cover** | clear sky | fully overcast |\n"
        "| **humidity** | bone dry | saturated air |"
    )
    cc = st.columns(2)
    with cc[0]:
        st.markdown("**👤 Inputs**")
        x1 = st.slider("cloud cover", 0.0, 1.0, 0.70, 0.05, key="w2a_x1")
        x2 = st.slider("humidity", 0.0, 1.0, 0.60, 0.05, key="w2a_x2")
    with cc[1]:
        for k, v in {"w2a_w1": 2.5, "w2a_w2": 1.5, "w2a_b": -2.0}.items():
            st.session_state.setdefault(k, v)
        st.markdown("**⚙️ Parameters** *(learned)*")
        w1 = st.slider("w₁ · cloud", -6.0, 6.0, step=0.1, key="w2a_w1")
        w2 = st.slider("w₂ · humidity", -6.0, 6.0, step=0.1, key="w2a_w2")
        b = st.slider("b · bias", -6.0, 6.0, step=0.1, key="w2a_b")
    st.button("↺ Defaults", key="w2a_reset", on_click=mv.set_state,
              args=({"w2a_w1": 2.5, "w2a_w2": 1.5, "w2a_b": -2.0},))

    act_name, act = mv.activation_picker("w2a_act", "sigmoid", "⚡ activation φ")

    x = np.array([x1, x2])
    w = np.array([w1, w2])
    z = float(w @ x + b)
    a = float(np.asarray(act["f"](np.array([z])), dtype=float)[0])

    st.markdown("#### 🧮 The arithmetic")
    st.latex(r"z = \mathbf w\cdot\mathbf x + b = " + mv.rowvec(w, 1) + mv.bmat(x) + " + (" + mv.fmt(b, 1) + ")")
    st.latex(f"z = ({w1:.1f})({x1:.2f}) + ({w2:.1f})({x2:.2f}) + ({b:.1f})"
             r"\;=\;" + f"{w1*x1:.2f} + {w2*x2:.2f} + ({b:.2f}) = {z:.3f}")
    st.latex(r"\varphi(z) = " + act["tex"])
    st.latex(r"a = \varphi(" + f"{z:.3f}" + ") = " + f"{a:.4f}")
    mv.activation_explainer(act_name, z_mark=z)

    m = st.columns(3)
    m[0].metric("z", f"{z:+.3f}")
    m[1].metric(f"a = {act_name}(z)", f"{a:.4f}" + (f"  ({a*100:.0f}%)" if act_name == "sigmoid" else ""))
    m[2].metric("decision (z > 0)", "TAKE IT ☔" if z > 0 else "LEAVE IT ☀️")

    st.markdown("#### 🗺️ The decision line — this is the *whole* neuron, drawn")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    gg = np.linspace(-0.02, 1.02, 200)
    GX, GY = np.meshgrid(gg, gg)
    P = np.asarray(act["f"](w1 * GX + w2 * GY + b), dtype=float)
    im = ax.contourf(GX, GY, P, levels=20, cmap="RdBu", alpha=0.85)
    ax.contour(GX, GY, w1 * GX + w2 * GY + b, levels=[0.0], colors="k", linewidths=2)
    ax.plot(x1, x2, "o", ms=12, color="#111827", zorder=5)
    ax.annotate(f"  you: {a*100:.0f}%", (x1, x2), fontsize=9, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9C9B95", alpha=0.9))
    ax.set_xlabel("cloud cover")
    ax.set_ylabel("humidity")
    ax.set_title("black line = the boundary  w·x + b = 0", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.colorbar(im, ax=ax, shrink=0.85, label=f"{act_name}(z)")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.info(
        f"**The line has an equation you can read off the weights:** $z=0$ means "
        f"${w1:.1f}\\,x_1 + {w2:.1f}\\,x_2 + ({b:.1f}) = 0$, i.e. "
        + (f"$x_2 = {-w1/w2:.2f}\\,x_1 + {-b/w2:.2f}$." if abs(w2) > 1e-9 else "a vertical line.")
        + " The **weight ratio sets the slope** (which feature matters more) and the **bias sets "
          "the intercept** (how much total evidence is needed). One neuron = one straight cut "
          "through feature space — no matter how many inputs it has.",
        icon=":material/lightbulb:")

# =========================================================================== #
# ② 4 INPUTS → 1 OUTPUT
# =========================================================================== #
with t2:
    st.markdown("#### 📋 Scenario — a spam filter with 4 signals")
    st.markdown(
        "More inputs change **nothing** structurally — the dot product just gets longer. But "
        "now we can no longer *draw* the boundary (it is a 3-D plane inside 4-D space), so we "
        "read the decision from the **contributions** instead. That is how real feature "
        "attribution works.\n\n"
        "| feature | `0.00` | `1.00` |\n|---|---|---|\n"
        "| **CAPS ratio** | no shouting | ALL CAPS |\n"
        "| **links** | no links | many links |\n"
        "| **\"free\" / \"winner\"** | absent | present |\n"
        "| **unknown sender** | in contacts | never seen |"
    )
    cc = st.columns(2)
    with cc[0]:
        st.markdown("**👤 This email**")
        x1 = st.slider("CAPS ratio", 0.0, 1.0, 0.40, 0.05, key="w2b_x1")
        x2 = st.slider("links", 0.0, 1.0, 0.60, 0.05, key="w2b_x2")
        x3 = st.slider('"free" / "winner"', 0.0, 1.0, 1.00, 0.05, key="w2b_x3")
        x4 = st.slider("unknown sender", 0.0, 1.0, 1.00, 0.05, key="w2b_x4")
    with cc[1]:
        for k, v in {"w2b_w1": 1.8, "w2b_w2": 2.2, "w2b_w3": 3.0, "w2b_w4": 1.2,
                     "w2b_b": -3.5}.items():
            st.session_state.setdefault(k, v)
        st.markdown("**⚙️ Parameters** *(learned)*")
        w1 = st.slider("w₁ · CAPS", -6.0, 6.0, step=0.1, key="w2b_w1")
        w2 = st.slider("w₂ · links", -6.0, 6.0, step=0.1, key="w2b_w2")
        w3 = st.slider("w₃ · spam words", -6.0, 6.0, step=0.1, key="w2b_w3")
        w4 = st.slider("w₄ · unknown sender", -6.0, 6.0, step=0.1, key="w2b_w4")
        b = st.slider("b · bias (how paranoid)", -8.0, 8.0, step=0.1, key="w2b_b")
    st.button("↺ Defaults", key="w2b_reset", on_click=mv.set_state,
              args=({"w2b_w1": 1.8, "w2b_w2": 2.2, "w2b_w3": 3.0, "w2b_w4": 1.2, "w2b_b": -3.5},))

    x = np.array([x1, x2, x3, x4])
    w = np.array([w1, w2, w3, w4])
    contrib = w * x
    z = float(contrib.sum() + b)
    a = float(mv.sigmoid(z))
    names = ["CAPS", "links", "spam words", "unknown sender"]

    st.markdown("#### 🧮 The arithmetic — a 4-term dot product")
    st.latex(r"z = \mathbf w\cdot\mathbf x + b = " + mv.rowvec(w, 1) + mv.bmat(x) + " + (" + mv.fmt(b, 1) + ")")
    st.latex(
        " + ".join(f"({w[i]:.1f})({x[i]:.2f})" for i in range(4)) + f" + ({b:.1f})"
        r"\;=\;" + " + ".join(f"{contrib[i]:.2f}" for i in range(4)) + f" + ({b:.2f}) = {z:.3f}"
    )
    st.latex(r"a = \sigma(" + f"{z:.3f}" + ") = " + f"{a:.4f}")

    m = st.columns(3)
    m[0].metric("z", f"{z:+.3f}")
    m[1].metric("P(spam)", f"{a*100:.1f}%")
    m[2].metric("verdict", "SPAM 🚫" if a > 0.5 else "INBOX ✅")

    st.markdown("**Which signal convicted it?**")
    st.pyplot(mv.contribution_chart(
        [f"{n}\n({w[i]:.1f} × {x[i]:.2f})" for i, n in enumerate(names)] + [f"bias\n({b:.1f})"],
        list(contrib) + [b], z), width="stretch")
    top = int(np.argmax(np.abs(contrib)))
    st.info(
        f"**Biggest single driver: {names[top]}** ({contrib[top]:+.2f} of the {z:+.3f} total). "
        "Note the bias is the *only* term that does not depend on the email — it is the filter's "
        "standing scepticism, and raising it alone would mark everything as spam. With 4 inputs "
        "the neuron still draws exactly **one flat boundary**, now a 3-D hyperplane in 4-D space: "
        "you cannot picture it, but the arithmetic is identical to the 2-input case.",
        icon=":material/lightbulb:")

# =========================================================================== #
# ③ 5 INPUTS → 2 OUTPUTS (one layer)
# =========================================================================== #
with t3:
    st.markdown("#### 📋 Scenario — hospital triage: routine or urgent?")
    st.markdown(
        "Five vital signs, **two** competing outputs. Two outputs means **two neurons**, so the "
        "weights become a **2×5 matrix** — one row per outcome. Softmax then turns the two "
        "scores into probabilities that sum to 1.\n\n"
        "| vital | `0.00` | `1.00` |\n|---|---|---|\n"
        "| **temperature** | normal | high fever |\n"
        "| **heart rate** | resting | tachycardic |\n"
        "| **blood pressure** | normal | dangerous |\n"
        "| **oxygen sat.** | critically low | perfect |\n"
        "| **pain** | none | severe |"
    )
    vitals = ["temp", "heart rate", "blood pr.", "oxygen", "pain"]
    st.markdown("**👤 This patient**")
    xc = st.columns(5)
    dx = [0.70, 0.60, 0.40, 0.30, 0.80]
    x = np.array([xc[i].slider(vitals[i], 0.0, 1.0, dx[i], 0.05, key=f"w2c_x{i}")
                  for i in range(5)])

    W_def = np.array([[-1.2, -1.0, -0.8, 1.5, -0.9],
                      [2.0, 1.8, 1.2, -2.2, 1.4]])
    b_def = np.array([2.0, -2.0])
    classes = ["routine", "urgent"]
    st.markdown("**⚙️ The 2×5 weight matrix + 2 biases — one row per outcome** *(learned)*")
    Wl = []
    for r in range(2):
        st.caption(f"row {r+1} → **{classes[r]}**")
        rc = st.columns(6)
        row = []
        for c in range(5):
            st.session_state.setdefault(f"w2c_w{r}{c}", float(W_def[r, c]))
            row.append(rc[c].number_input(f"w[{r+1},{c+1}]", -4.0, 4.0, step=0.1,
                                          key=f"w2c_w{r}{c}", format="%.1f"))
        st.session_state.setdefault(f"w2c_b{r}", float(b_def[r]))
        bb = rc[5].number_input(f"b[{r+1}]", -6.0, 6.0, step=0.1, key=f"w2c_b{r}", format="%.1f")
        Wl.append(row + [bb])
    W = np.array([r[:5] for r in Wl])
    b = np.array([r[5] for r in Wl])
    st.button("↺ Reset all 12", key="w2c_reset", on_click=mv.set_state, args=(
        {f"w2c_w{r}{c}": float(W_def[r, c]) for r in range(2) for c in range(5)}
        | {f"w2c_b{r}": float(b_def[r]) for r in range(2)},))

    z = W @ x + b
    p = mv.softmax(z)

    st.markdown("#### 🧮 Step 1 — a 2×5 matrix times a 5-vector")
    st.latex(r"\mathbf z = W\mathbf x + \mathbf b = " + mv.bmat(W, 1) + mv.bmat(x) + " + " + mv.bmat(b, 1))
    for r in range(2):
        st.latex(
            f"z_{r+1} = " + " + ".join(f"({W[r,c]:.1f})({x[c]:.2f})" for c in range(5)) +
            f" + ({b[r]:.1f}) = {z[r]:.3f}" + r"\quad\text{(" + classes[r] + ")}"
        )
    st.latex(r"\mathbf z = " + mv.bmat(z, 3) + r",\qquad \text{shapes: }(2\times5)(5\times1)+(2\times1)=(2\times1)")

    st.markdown("#### 🧮 Step 2 — softmax over **two** scores")
    st.latex(r"p_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}},\qquad p_2 = 1-p_1")
    st.latex(r"\mathbf p = " + mv.rowvec(p, 4) + r",\qquad \text{margin } z_2-z_1 = " + f"{z[1]-z[0]:+.3f}")

    m = st.columns(3)
    m[0].metric("routine", f"{p[0]*100:.1f}%")
    m[1].metric("urgent", f"{p[1]*100:.1f}%")
    m[2].metric("triage", classes[int(p.argmax())].upper())
    (st.error if p.argmax() == 1 else st.success)(
        f"**{classes[int(p.argmax())].upper()}** — {p.max()*100:.1f}% confident.",
        icon=":material/emergency:")

    st.markdown("**How each vital pushed each outcome** (per-row contributions $w_{rc}x_c$):")
    gc = st.columns(2)
    for r in range(2):
        with gc[r]:
            st.pyplot(mv.contribution_chart(
                [f"{vitals[c]}" for c in range(5)] + ["bias"],
                list(W[r] * x) + [b[r]], float(z[r]),
                xlabel=f"contribution to z ({classes[r]})", height=2.9), width="stretch")

    with st.expander("📖 Why two outputs (and not one)?"):
        st.markdown(
            "With **two mutually exclusive classes** you could use a *single* sigmoid neuron — "
            "and in fact that is equivalent: softmax over two scores depends only on the "
            "**difference** $z_2-z_1$, and $p_2=\\sigma(z_2-z_1)$. Check it against the margin "
            f"above: $\\sigma({z[1]-z[0]:.3f}) = {mv.sigmoid(z[1]-z[0]):.4f} = p_2$. So the "
            "2-output form carries one redundant degree of freedom.\n\n"
            "We still write it with two rows because it **generalises**: 3, 10, or 50 000 classes "
            "(an LLM's vocabulary!) all use exactly this shape — one row of $W$ per class, "
            "softmax over the scores, cross-entropy as the loss (Math **X5**). The 5→2 layer here "
            "holds $2\\times5+2=\\mathbf{12}$ learned numbers.\n\n"
            "**Notice what it still cannot do:** each output is a *straight-line* function of the "
            "vitals. It cannot express *'high fever **and** low oxygen is an emergency, but "
            "either one alone is fine'* — that interaction needs a hidden layer, which is tab ④."
        )

# =========================================================================== #
# ④ 5 → 3 → 2 (two layers)
# =========================================================================== #
with t4:
    st.markdown("#### 📋 Scenario — the same triage, now with a hidden layer")
    st.markdown(
        "Same five vitals, same two outcomes — but now the signal passes through **3 hidden "
        "units** first. Each hidden unit is a **syndrome detector**: a learned combination of "
        "vitals that fires only when a pattern is present (ReLU, so it is silent below "
        "threshold). The output layer then reasons over *syndromes* instead of raw vitals."
    )
    st.markdown("**👤 This patient**")
    xc = st.columns(5)
    dx = [0.70, 0.60, 0.40, 0.30, 0.80]
    x = np.array([xc[i].slider(vitals[i], 0.0, 1.0, dx[i], 0.05, key=f"w2d_x{i}")
                  for i in range(5)])

    W1 = np.array([[1.6, 0.4, 0.0, 0.0, 1.2],
                   [0.0, 1.8, 1.5, 0.0, 0.6],
                   [0.3, 0.5, 0.0, -2.0, 0.4]])
    b1 = np.array([-1.4, -1.5, -0.3])
    hidden_names = ["infection", "cardiac", "respiratory"]

    st.markdown("**🧠 Layer 1 — three syndrome detectors** *(fixed here so they stay readable; "
                "in a real net all 15 weights + 3 biases are learned)*")
    st.latex(r"W^{(1)} = " + mv.bmat(W1, 1) + r"\;(3\times5),\qquad \mathbf b^{(1)} = " + mv.bmat(b1, 1))
    st.caption("Row 1 watches temperature + pain → *infection*. Row 2 watches heart rate + blood "
               "pressure → *cardiac*. Row 3 fires when oxygen drops (note the −2.0) → *respiratory*.")

    hact_name, hact = mv.activation_picker(
        "w2d_act", "ReLU", "⚡ hidden-layer activation φ — try **linear** to see the net collapse")
    z1 = W1 @ x + b1
    h = np.asarray(hact["f"](z1), dtype=float)

    st.markdown(f"#### 🧮 Step 1 — hidden layer  $\\mathbf h = \\varphi(W^{{(1)}}\\mathbf x + \\mathbf b^{{(1)}})$  with $\\varphi=${hact_name}")
    for r in range(3):
        st.latex(
            f"z^{{(1)}}_{r+1} = " + " + ".join(f"({W1[r,c]:.1f})({x[c]:.2f})" for c in range(5)) +
            f" + ({b1[r]:.1f}) = {z1[r]:.3f}" +
            r"\;\Rightarrow\; h_" + str(r + 1) + r" = \varphi(" + f"{z1[r]:.3f}" + f") = {h[r]:.3f}"
        )
    st.latex(r"\mathbf h = " + mv.bmat(h, 3))
    hc = st.columns(3)
    for r in range(3):
        hc[r].metric(f"h{r+1} · {hidden_names[r]}",
                     f"{h[r]:.3f}", "firing" if h[r] > 0 else "silent",
                     delta_color="inverse" if h[r] > 0 else "off")

    mv.activation_explainer(hact_name, z_mark=float(z1[0]))

    W2_def = np.array([[-1.5, -1.4, -1.6], [2.0, 1.9, 2.2]])
    b2_def = np.array([1.2, -1.0])
    st.markdown("**⚙️ Layer 2 — how to weigh the syndromes** *(yours to set)*")
    W2l = []
    for r in range(2):
        st.caption(f"row {r+1} → **{classes[r]}**")
        rc = st.columns(4)
        row = []
        for c in range(3):
            st.session_state.setdefault(f"w2d_w{r}{c}", float(W2_def[r, c]))
            row.append(rc[c].number_input(f"{hidden_names[c]}", -4.0, 4.0, step=0.1,
                                          key=f"w2d_w{r}{c}", format="%.1f"))
        st.session_state.setdefault(f"w2d_b{r}", float(b2_def[r]))
        row.append(rc[3].number_input(f"bias b[{r+1}]", -6.0, 6.0, step=0.1,
                                      key=f"w2d_b{r}", format="%.1f"))
        W2l.append(row)
    W2 = np.array([r[:3] for r in W2l])
    b2 = np.array([r[3] for r in W2l])
    st.button("↺ Reset layer 2", key="w2d_reset", on_click=mv.set_state, args=(
        {f"w2d_w{r}{c}": float(W2_def[r, c]) for r in range(2) for c in range(3)}
        | {f"w2d_b{r}": float(b2_def[r]) for r in range(2)},))

    z2 = W2 @ h + b2
    p = mv.softmax(z2)

    st.markdown("#### 🧮 Step 2 — output layer  $\\mathbf z^{(2)} = W^{(2)}\\mathbf h + \\mathbf b^{(2)}$")
    st.latex(r"\mathbf z^{(2)} = " + mv.bmat(W2, 1) + mv.bmat(h, 3) + " + " + mv.bmat(b2, 1))
    for r in range(2):
        st.latex(
            f"z^{{(2)}}_{r+1} = " + " + ".join(f"({W2[r,c]:.1f})({h[c]:.3f})" for c in range(3)) +
            f" + ({b2[r]:.1f}) = {z2[r]:.3f}" + r"\quad\text{(" + classes[r] + ")}"
        )
    st.latex(r"\mathbf p = \mathrm{softmax}(\mathbf z^{(2)}) = " + mv.rowvec(p, 4))

    m = st.columns(3)
    m[0].metric("routine", f"{p[0]*100:.1f}%")
    m[1].metric("urgent", f"{p[1]*100:.1f}%")
    m[2].metric("triage", classes[int(p.argmax())].upper())
    (st.error if p.argmax() == 1 else st.success)(
        f"**{classes[int(p.argmax())].upper()}** — {p.max()*100:.1f}% confident, driven by the "
        + (", ".join(f"**{hidden_names[i]}**" for i in np.argsort(-h)[:2] if h[i] > 0) or "**no**")
        + " signal.", icon=":material/emergency:")

    if hact_name == "linear":
        Weff = W2 @ W1
        beff = W2 @ b1 + b2
        st.error(
            "**⚠ You just collapsed the network.** With a *linear* hidden layer, these two "
            "layers are algebraically identical to **one** 5→2 layer — the 3 hidden units bought "
            "you nothing. Here is the single equivalent matrix:", icon=":material/warning:")
        st.latex(r"W_{\text{eff}} = W^{(2)}W^{(1)} = " + mv.bmat(Weff, 2) + r"\;(2\times5)"
                 r",\qquad \mathbf b_{\text{eff}} = W^{(2)}\mathbf b^{(1)} + \mathbf b^{(2)} = "
                 + mv.bmat(beff, 2))
        st.latex(r"W_{\text{eff}}\mathbf x + \mathbf b_{\text{eff}} = " + mv.bmat(Weff @ x + beff, 3)
                 + r"\;=\;\mathbf z^{(2)} = " + mv.bmat(z2, 3) + r"\quad\checkmark")
        st.markdown(
            "Identical outputs, from **12** parameters instead of 26. Switch back to **ReLU** "
            "and the equality breaks instantly — because $\\max(0,\\cdot)$ cannot be absorbed "
            "into a matrix product. *This is the single best argument for why the nonlinearity "
            "exists* (neuron lesson §8)."
        )

    st.markdown("#### 🔢 The full shape ledger")
    st.markdown(
        "| step | operation | shapes | parameters |\n|---|---|---|---|\n"
        "| layer 1 | $W^{(1)}\\mathbf x + \\mathbf b^{(1)}$ | $(3\\times5)(5\\times1)+(3\\times1)=(3\\times1)$ | $15+3=18$ |\n"
        "| ReLU | element-wise | $(3\\times1)$ | 0 |\n"
        "| layer 2 | $W^{(2)}\\mathbf h + \\mathbf b^{(2)}$ | $(2\\times3)(3\\times1)+(2\\times1)=(2\\times1)$ | $6+2=8$ |\n"
        "| softmax | element-wise | $(2\\times1)$ | 0 |\n"
        "| | | **total** | **26** |"
    )
    with st.expander("📖 What the hidden layer actually bought us"):
        st.markdown(
            "**1 · Non-linear interactions.** ReLU is the whole point. A hidden unit stays at "
            "**exactly 0** until its evidence crosses threshold, then rises linearly — so it "
            "encodes *'only once this pattern is present'*. The output can then say *'urgent if "
            "the **respiratory** detector fires, whatever the temperature says'*. The one-layer "
            "model in tab ③ can never express that: each of its outputs is a single flat plane "
            "over the raw vitals.\n\n"
            "**2 · A change of representation.** Layer 1 rewrites the patient from 5 raw numbers "
            "into 3 *meaningful* ones. That is the same move as the family-tree net (Embeddings) "
            "and the CNN hierarchy (edges → parts → objects): **each layer's features are built "
            "from the layer below**, and only the last layer does the actual classifying.\n\n"
            "**3 · A composition of matrices.** Note that without the ReLU the whole thing would "
            "collapse: $W^{(2)}(W^{(1)}\\mathbf x)=(W^{(2)}W^{(1)})\\mathbf x$ is just *one* "
            "$2\\times5$ matrix — two layers with no nonlinearity are exactly equivalent to one "
            "layer (neuron lesson §8). **The nonlinearity is what makes depth worth anything.**\n\n"
            "**4 · The cost.** 12 parameters became **26**. Real nets take this trade billions of "
            "times over — and every one of those numbers is found by the same gradient-descent "
            "step you did by hand on the previous page."
        )
