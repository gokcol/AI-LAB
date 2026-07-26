"""RNN — recurrent networks for sequences (ANN module, roadmap e15 / Tier 3).

Sequences need memory. An RNN walks a sequence one step at a time, carrying a hidden
**state** that summarizes the past. The Live tab runs a one-unit RNN,
h_t = tanh(w_x·x_t + w_h·h_{t-1}), so you can watch the **recurrent weight w_h set how long
it remembers** — the seed of both long-range learning and the vanishing-gradient problem.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons


def _sequence(kind, n):
    x = np.zeros(n)
    if kind == "Impulse (one spike)":
        x[1] = 1.0
    elif kind == "Step (turns on)":
        x[n // 3:] = 1.0
    elif kind == "Alternating":
        x[::2] = 1.0
    else:  # Random
        x = (np.random.default_rng(0).random(n) < 0.3).astype(float)
    return x


def _run_rnn(x, wx, wh, b=0.0):
    h, hs = 0.0, []
    for xt in x:
        h = np.tanh(wx * xt + wh * h + b)
        hs.append(h)
    return np.array(hs)


_DIAG_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 720 230" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="An RNN drawn two ways: folded as one cell whose hidden state feeds back into itself (recurrent weight Wh), with input weight Wx and output weight Wy; and unrolled over time as a chain h0 to h3 where each step takes an input x_t, updates the state, and emits y_t."><defs><marker id="rnn" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="95" y="26" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#33312E">folded (one cell, reused)</text><circle cx="95" cy="120" r="26" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.8"/><text x="95" y="125" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#5A3E14">h</text><path d="M118,104 C160,80 160,160 118,136" fill="none" stroke="#9A6A2A" stroke-width="1.8" marker-end="url(#rnn)"/><text x="158" y="124" font-family="sans-serif" font-size="11" fill="#9A6A2A">Wₕ</text><line x1="95" y1="180" x2="95" y2="148" stroke="#5B8FC2" stroke-width="1.8" marker-end="url(#rnn)"/><text x="80" y="174" font-family="sans-serif" font-size="11" fill="#0C447C">Wₓ</text><text x="95" y="198" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">x</text><line x1="95" y1="92" x2="95" y2="62" stroke="#1D9E75" stroke-width="1.8" marker-end="url(#rnn)"/><text x="80" y="84" font-family="sans-serif" font-size="11" fill="#0E5E45">Wy</text><text x="95" y="56" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0E5E45">y</text><line x1="180" y1="120" x2="208" y2="120" stroke="#9C9B95" stroke-width="1.5" stroke-dasharray="4 3"/><text x="430" y="26" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#33312E">unrolled over time (same weights every step)</text><g font-family="sans-serif" font-size="13" text-anchor="middle"><circle cx="280" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="280" y="125" fill="#5A3E14">h₁</text><circle cx="380" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="380" y="125" fill="#5A3E14">h₂</text><circle cx="480" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="480" y="125" fill="#5A3E14">h₃</text><circle cx="580" cy="120" r="22" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="1.6"/><text x="580" y="125" fill="#5A3E14">h₄</text></g><g stroke="#9A6A2A" stroke-width="1.7" fill="none"><line x1="234" y1="120" x2="256" y2="120" marker-end="url(#rnn)"/><line x1="302" y1="120" x2="356" y2="120" marker-end="url(#rnn)"/><line x1="402" y1="120" x2="456" y2="120" marker-end="url(#rnn)"/><line x1="502" y1="120" x2="556" y2="120" marker-end="url(#rnn)"/></g><text x="234" y="112" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#9A6A2A">Wₕ</text><g stroke="#5B8FC2" stroke-width="1.6" fill="none"><line x1="280" y1="170" x2="280" y2="144" marker-end="url(#rnn)"/><line x1="380" y1="170" x2="380" y2="144" marker-end="url(#rnn)"/><line x1="480" y1="170" x2="480" y2="144" marker-end="url(#rnn)"/><line x1="580" y1="170" x2="580" y2="144" marker-end="url(#rnn)"/></g><g stroke="#1D9E75" stroke-width="1.6" fill="none"><line x1="280" y1="96" x2="280" y2="70" marker-end="url(#rnn)"/><line x1="380" y1="96" x2="380" y2="70" marker-end="url(#rnn)"/><line x1="480" y1="96" x2="480" y2="70" marker-end="url(#rnn)"/><line x1="580" y1="96" x2="580" y2="70" marker-end="url(#rnn)"/></g><g font-family="sans-serif" font-size="10" text-anchor="middle" fill="#0C447C"><text x="280" y="184">x₁</text><text x="380" y="184">x₂</text><text x="480" y="184">x₃</text><text x="580" y="184">x₄</text></g><g font-family="sans-serif" font-size="10" text-anchor="middle" fill="#0E5E45"><text x="280" y="64">y₁</text><text x="380" y="64">y₂</text><text x="480" y="64">y₃</text><text x="580" y="64">y₄</text></g><text x="430" y="212" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">the hidden state h carries information forward — the network's memory</text></svg></div>'''


_THEORY = r"""
## 1. Sequences need memory

MLPs and CNNs take a **fixed-size** input. But language, speech, music and time-series are
**variable-length** and **order matters** ("dog bites man" ≠ "man bites dog"). A **recurrent
neural network (RNN)** handles this by walking the sequence **one step at a time**, carrying
a hidden **state** vector that summarizes everything seen so far — its **memory**.

## 2. The recurrence

At each step the RNN folds the new input into the running state:
$$ h_t = \tanh\!\big(W_x\,x_t + W_h\,h_{t-1} + b\big),\qquad y_t = W_y\,h_t. $$
The **same** weights $W_x, W_h, W_y$ are used at *every* time step (**weight sharing across
time**, like a CNN shares a kernel across space). $h_{t-1}$ is what makes it recurrent: the
output depends on the whole history, not just the current input.

## 3. Unrolling & backprop-through-time

"Unroll" the loop across the sequence and an RNN becomes a **deep feed-forward network that
shares weights** between layers. You train it with ordinary backprop applied to that
unrolled graph — called **backprop-through-time (BPTT)**: gradients flow backward across the
time steps (Backprop page).

<DIAG/>

## 4. The recurrent weight is the memory knob

How long does information last? It's governed by $W_h$. Feed a single spike (the demo's
**impulse**) and watch the state:
- **small $W_h$** → the state forgets almost immediately (short memory);
- **$W_h \approx 1$** → it persists for many steps (long memory);
- **$W_h > 1$** → it can blow up (tanh caps it here, but gradients won't be capped).

Slide $w_h$ in the Live tab and see the impulse response stretch or vanish.

## 5. Vanishing & exploding through time

Backprop multiplies by $W_h$ at **every** step, so over a long sequence the gradient is
$W_h$ raised to a large power (Backprop §10). $|W_h|<1$ → it **vanishes** (the RNN can't
learn long-range dependencies — it forgets the start of the sentence); $|W_h|>1$ → it
**explodes**. This is the central weakness of plain RNNs.

## 6. LSTM — gated memory, in full

The fix is **gates**. An **LSTM** (Hochreiter & Schmidhuber, 1997) adds a second, protected
memory — the **cell state** $c_t$, a "conveyor belt" running straight through time — and three
learned gates that decide what happens to it. Each gate is nothing but a **sigmoid neuron**
whose output in $(0,1)$ acts as a *valve*, multiplied element-wise ($\odot$):

$$ f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),\quad i_t=\sigma(W_i[x_t,h_{t-1}]+b_i),\quad o_t=\sigma(W_o[x_t,h_{t-1}]+b_o) $$
$$ \tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c),\qquad c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\qquad h_t=o_t\odot\tanh(c_t) $$

Read it as three decisions: **forget** — how much of the old memory to keep; **input** — how
much of the new candidate to write; **output** — how much of the memory to expose as this
step's hidden state. Note the LSTM has **four** weight matrices where a vanilla RNN has one,
so it costs ~4× the parameters.

**Why this actually fixes the gradient.** Look at the boxed update: the cell is changed by
**addition**, not by repeated multiplication with a weight matrix. Differentiating gives
$$ \frac{\partial c_t}{\partial c_{t-1}} = f_t. $$
If the network learns to hold the forget gate **open** ($f_t\approx1$), the gradient flows
back through time **undamped** — the *constant error carousel*. Compare with the vanilla
RNN's $w_h\tanh'$, which is strictly less than 1 in practice. That single change is what
made long-range sequence learning work. *(A practical trick that follows: initialise the
forget-gate bias **positive** — often 1.0 — so the cell starts out remembering by default.)*

## 7. GRU — the lighter alternative

The **Gated Recurrent Unit** (Cho et al., 2014) merges the cell and hidden state and uses
**two** gates instead of three:
$$ z_t=\sigma(W_z[x_t,h_{t-1}]),\quad r_t=\sigma(W_r[x_t,h_{t-1}]),\quad \tilde h_t=\tanh(W[x_t,\;r_t\odot h_{t-1}]) $$
$$ h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t $$
The **update** gate $z_t$ does the job of forget *and* input with one knob (keep $1-z$ of the
old, write $z$ of the new); the **reset** gate $r_t$ decides how much history is visible when
forming the candidate. ~25 % fewer parameters and faster, with accuracy usually
indistinguishable from an LSTM — so GRU is a sensible default when data or compute is tight.

## 8. Sequence shapes — what RNNs are wired to do

The same cell serves several task shapes:

| shape | example | wiring |
|---|---|---|
| **one → many** | image → caption | one input, emit a sequence |
| **many → one** | review → sentiment | read it all, use the final $h_T$ |
| **many → many (aligned)** | tagging each word | one output per step |
| **many → many (seq2seq)** | translation | encoder reads, decoder writes |

**Seq2seq / encoder–decoder** (2014) is the important one: an **encoder** RNN reads the whole
source sentence into a final state, and a **decoder** RNN generates the target from it. This
was the state of the art in translation — and it exposed the flaw that created modern AI: the
entire source sentence has to be squeezed into **one fixed-size vector**, an *information
bottleneck* that gets worse the longer the sentence. **Attention** was invented in 2014–15
precisely to fix this, by letting the decoder look back at *all* encoder states instead of
just the last one. That patch eventually ate the whole architecture (§11).

## 9. Deeper and bidirectional

Two standard upgrades. **Stacked (deep) RNNs** feed one recurrent layer's outputs into another
as its inputs, building hierarchy in the same way stacked MLP layers do. **Bidirectional
RNNs** run one pass left→right and another right→left, concatenating the two states — so each
position sees the *whole* sentence. Bidirectional is excellent for **understanding** tasks
(tagging, classification — the idea behind BERT's "B") but impossible for **generation**,
where the future is not available yet.

## 10. Training them in practice

Three techniques you will always meet:

- **Truncated BPTT** — unrolling 10 000 steps would exhaust memory, so backprop is cut off
  after a window (say 50–200 steps). The forward state still carries further than the
  gradient does.
- **Gradient clipping** — rescale the gradient when its norm exceeds a threshold. This is the
  standard, cheap cure for the *exploding* half of the problem (the vanishing half needs
  gates). See Math **X4 §18**.
- **Teacher forcing** — during training, feed the *true* previous token rather than the
  model's own prediction. It converges much faster, but creates **exposure bias**: at
  generation time the model must consume its own (imperfect) outputs, a distribution it never
  trained on.

## 11. Why Transformers replaced them (mostly)

RNNs have two stubborn problems. They are **inherently sequential** — step $t$ needs step
$t{-}1$, so the time dimension cannot be parallelized, which wastes a GPU and makes training
on long corpora slow. And even a gated cell strains at *very* long range: the path between two
distant tokens is still $O(\text{distance})$ steps long.

**Attention / Transformers** (Attention page) fix both at once: every position is computed
**in parallel**, and any two tokens are connected **directly in one step** — path length
$O(1)$ instead of $O(n)$. The trade is cost: attention is $O(n^2)$ in sequence length where an
RNN is $O(n)$ with $O(1)$ memory per step.

That trade is why RNNs are **not** extinct. They still win where input arrives **streaming**,
where latency and memory are tight (on-device speech, control loops), and for modest-length
time-series. It is also why the **state-space models** of recent years (S4, Mamba) are
interesting: they are recurrent at inference — $O(1)$ per token, no growing KV-cache — while
being parallelizable during training. The recurrent idea keeps coming back.

*(Roadmap e15; the real version is an LSTM char-model trained in PyTorch.)*
"""

_QUIZ = [
    lessons.Question(
        "What makes an RNN suited to sequences?",
        ["it has more layers", "it carries a hidden state that summarizes the past, updated step by step with shared weights",
         "it uses convolution", "it ignores order"], 1,
        "h_t = tanh(W_x x_t + W_h h_{t-1}) — the state is memory, and the same weights apply at every step."),
    lessons.Question(
        "In the demo, the recurrent weight w_h controls…",
        ["the input size", "how long information persists in the state (memory length)",
         "the number of classes", "the learning rate"], 1,
        "Small w_h forgets fast; w_h≈1 remembers long; w_h>1 blows up — it's the memory knob."),
    lessons.Question(
        "Why do plain RNNs struggle with long-range dependencies?",
        ["too few parameters", "gradients get multiplied by W_h each step, so they vanish or explode over long sequences",
         "they can't use tanh", "the inputs are too short"], 1,
        "Repeated multiplication across time (Backprop §10) shrinks or blows up gradients — the vanishing/exploding problem."),
    lessons.Question(
        "What do LSTM gates add over a plain RNN?",
        ["nothing", "a protected cell state + forget/input/output gates that preserve long-term memory",
         "convolution", "a softmax"], 1,
        "Gated, mostly-additive cell-state updates let gradients (and memory) survive far longer."),
    lessons.Question(
        "Transformers largely replaced RNNs because they…",
        ["are smaller", "process all positions in parallel and connect any two tokens directly (no sequential bottleneck)",
         "don't need data", "use no attention"], 1,
        "Parallelism + direct long-range connections beat the RNN's step-by-step state for most NLP."),
]

_TASKS = r"""
### In the Live tab
1. Use **Impulse** input. At **w_h = 0.2**, how many steps until the state ≈ 0? Now **w_h =
   0.9** — count again. That's *memory length* as a function of the recurrent weight.
2. Push **w_h > 1** with a **Step** input — watch the state pin to the tanh limits (and
   imagine the *gradient*, which tanh does **not** cap → exploding).
3. Set **w_h ≈ 0** — the state just copies the (scaled) input: no memory, an RNN reduced to
   a per-step neuron.

### Concept
4. Write the unrolled computation for a length-3 sequence and mark where the *same* W_h is
   reused — then explain why that causes vanishing/exploding gradients.
5. In one line each: what do the LSTM **forget**, **input**, and **output** gates do?
6. Give one task where you'd still pick an RNN/LSTM over a Transformer, and why.
"""

_REFS = r"""
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory (LSTM)*.
- Cho et al. (2014) — *GRU*. · Karpathy (2015) — *The Unreasonable Effectiveness of RNNs*.
- Olah — *Understanding LSTM Networks* (the classic gate walkthrough).
- Vaswani et al. (2017) — *Attention Is All You Need* (why RNNs were superseded).
- In this lab: **Backprop** (BPTT / vanishing gradients), **Attention** (the replacement),
  **Tiny GPT**.
"""


st.title("RNN — recurrent networks for sequences")
st.caption("Walk a sequence carrying a hidden state. The Live tab runs a 1-unit RNN so you "
           "can watch the recurrent weight set how long it remembers.")

lessons.predict(
    'Feed one spike, then zeros. With recurrent weight **wₕ ≈ 1** vs **wₕ ≈ 0.3**, which remembers the spike longer — and what happens if **wₕ > 1**?',
    '**wₕ ≈ 1** holds the value for many steps (long memory); **wₕ ≈ 0.3** forgets fast (×0.3 each step); **wₕ > 1** blows up. The same wₕ multiplies the *gradient* every step too — exactly why long sequences vanish/explode, and why LSTMs add gates.',
)

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔁 Hidden state", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _hidden_state():
    st.markdown("A one-unit RNN: $h_t = \\tanh(w_x\\,x_t + w_h\\,h_{t-1})$. Pick an input "
                "sequence and tune the weights — watch the **hidden state** (memory) evolve:")
    c = st.columns(3)
    kind = c[0].selectbox("input sequence", ["Impulse (one spike)", "Step (turns on)",
                                             "Alternating", "Random"], key="rnn_in")
    wx = c[1].slider("input weight wₓ", 0.0, 3.0, 1.0, 0.1, key="rnn_wx")
    wh = c[2].slider("recurrent weight wₕ (memory)", 0.0, 1.3, 0.7, 0.05, key="rnn_wh")
    n = 18
    x = _sequence(kind, n)
    h = _run_rnn(x, wx, wh)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    t = np.arange(n)
    ax.bar(t, x, width=0.5, color="#CFE3F5", label="input $x_t$")
    ax.plot(t, h, "-o", color="#9A6A2A", lw=1.8, ms=4, label="hidden state $h_t$")
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("time step t"); ax.set_ylim(-1.1, 1.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("the state carries the past forward", fontsize=10)
    st.pyplot(fig, width="stretch")

    if kind.startswith("Impulse"):
        decayed = next((i - 1 for i in range(2, n) if abs(h[i]) < 0.1 * abs(h[1]) + 1e-9), n)
        st.metric("impulse remembered for ≈", f"{max(decayed-1,0)} steps",
                  help="how long after the spike the state stays above 10% — i.e. memory length")
    st.info("The recurrent weight wₕ is the **memory knob**: small forgets fast, ≈1 remembers "
            "long, >1 blows up. The same wₕ multiplies the gradient at every step too — which "
            "is exactly why long sequences vanish/explode, and why LSTMs add gates.",
            icon=":material/history:")


def _bptt():
    st.markdown(
        "**Why long-range learning fails.** To learn that step $t$ mattered for the loss at step "
        "$T$, the gradient must survive the whole journey back. Each step multiplies it by the "
        "*same* two factors — the recurrent weight and the activation's slope:")
    st.latex(r"\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T} w_h\,\tanh'(z_k)"
             r"\;\approx\;\big(w_h\,\tanh'\big)^{T-t}")
    st.markdown("A **product of the same number** repeated — so it is exponential in the lag: "
                "below 1 it vanishes and well above 1 it explodes, and the band in between is narrow. Measured on this demo at lag 30: $w_h=0.5$ gives 9.0e-10, $w_h=1.5$ gives 2.8e-11 — but $w_h=1.0$ gives 0.21, which survives. The knife edge exists; it is just thin, and its position depends on the data.")
    whs = st.multiselect("recurrent weights to race", [0.5, 0.7, 0.9, 1.0, 1.1, 1.3],
                         default=[0.5, 0.9, 1.0, 1.1], key="rnn_bptt_w")
    maxlag = st.slider("maximum lag (steps back in time)", 5, 60, 30, 5, key="rnn_bptt_lag")

    if whs:
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        rows = []
        for wh in whs:
            g, hh, curve = 1.0, 0.3, []
            for _ in range(maxlag):
                z = wh * hh
                g *= wh * (1 - np.tanh(z) ** 2)
                hh = np.tanh(z)
                curve.append(abs(g))
            ax.semilogy(range(1, maxlag + 1), np.maximum(curve, 1e-300), lw=2.1, marker="o",
                        ms=2.6, label=f"wₕ = {wh}")
            verdict = ("🔴 vanished" if curve[-1] < 1e-4 else
                       ("🟢 survives" if curve[-1] > 1e-2 else "🟡 fading"))
            rows.append(f"| **wₕ = {wh}** | {curve[0]:.3f} | {curve[-1]:.2e} | {verdict} |")
        ax.axhline(1.0, color="0.7", ls=":", lw=1)
        ax.set_xlabel("lag  T − t  (how far back the gradient must travel)", fontsize=9)
        ax.set_ylabel("|∂h_T/∂h_t|  (log)", fontsize=9)
        ax.set_title("gradient surviving back through time", fontsize=9.5)
        ax.legend(fontsize=8); ax.tick_params(labelsize=8)
        fig.tight_layout(); st.pyplot(fig, width="stretch")
        st.markdown("| recurrent weight | gradient at lag 1 | at max lag | verdict |\n"
                    "|---|---|---|---|\n" + "\n".join(rows))
        st.info(
            "**The safe band is narrow, not empty.** At $w_h=1$ the $\\tanh'$ factor is < 1 so the "
            "gradient still decays — but slowly: 0.21 at lag 30 here, which is usable. Move a "
            "little either way and it goes: 0.030 at $w_h=0.9$, 0.014 at $w_h=1.1$, and by "
            "$w_h=0.5$ or $1.5$ it is ~1e-10. Push $w_h$ well above 1 and it grows until tanh "
            "saturates and then collapses anyway. That is the **vanishing/exploding gradient "
            "through time**: not the absence of a working setting, but a knife edge you would "
            "have to balance on for every dataset — which is not engineering. And "
            "it is why a plain RNN struggles to connect events more than ~10–20 steps apart. "
            "*Exploding* is the easy half — just **clip** the gradient. *Vanishing* needs an "
            "architectural fix: **gates**.", icon=":material/trending_down:")


def _lstm_cell():
    st.markdown(
        "**The fix: let the network decide what to keep.** An LSTM adds a second, protected "
        "memory — the **cell state** $c_t$ — and three learned **gates** that control it. Each "
        "gate is just a sigmoid neuron outputting a number in $(0,1)$: a *valve*.")
    st.latex(r"f_t=\sigma(W_f\cdot[x_t,h_{t-1}]),\quad i_t=\sigma(W_i\cdot[x_t,h_{t-1}]),"
             r"\quad o_t=\sigma(W_o\cdot[x_t,h_{t-1}])")
    st.latex(r"\tilde c_t=\tanh(W_c\cdot[x_t,h_{t-1}]),\qquad "
             r"\boxed{c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t},\qquad h_t=o_t\odot\tanh(c_t)")
    c = st.columns(3)
    x_t = c[0].slider("input xₜ", -2.0, 2.0, 1.0, 0.1, key="ls_x")
    h_p = c[1].slider("previous hidden hₜ₋₁", -1.0, 1.0, 0.3, 0.1, key="ls_h")
    c_p = c[2].slider("previous cell cₜ₋₁", -2.0, 2.0, 0.5, 0.1, key="ls_c")
    st.caption("gate weights — each row is (weight on x, weight on h, bias)")
    g = st.columns(4)
    W = {}
    defs = {"f (forget)": (0.8, 0.5, -0.3), "i (input)": (1.2, -0.4, 0.2),
            "o (output)": (0.6, 0.9, 0.1), "c̃ (candidate)": (1.5, 0.7, -0.2)}
    for j, (nm, d) in enumerate(defs.items()):
        with g[j]:
            st.markdown(f"**{nm}**")
            W[nm] = (
                st.number_input("wx", -3.0, 3.0, d[0], 0.1, key=f"ls_wx{j}", format="%.1f"),
                st.number_input("wh", -3.0, 3.0, d[1], 0.1, key=f"ls_wh{j}", format="%.1f"),
                st.number_input("b", -3.0, 3.0, d[2], 0.1, key=f"ls_b{j}", format="%.1f"),
            )

    def pre(nm):
        w = W[nm]
        return w[0] * x_t + w[1] * h_p + w[2]

    f = float(_sigmoid(pre("f (forget)")))
    i = float(_sigmoid(pre("i (input)")))
    o = float(_sigmoid(pre("o (output)")))
    cand = float(np.tanh(pre("c̃ (candidate)")))
    c_t = f * c_p + i * cand
    h_t = o * float(np.tanh(c_t))

    m = st.columns(4)
    m[0].metric("f · forget", f"{f:.4f}", "keep " + f"{f*100:.0f}%", delta_color="off")
    m[1].metric("i · input", f"{i:.4f}", "write " + f"{i*100:.0f}%", delta_color="off")
    m[2].metric("o · output", f"{o:.4f}", "expose " + f"{o*100:.0f}%", delta_color="off")
    m[3].metric("c̃ · candidate", f"{cand:+.4f}")
    st.latex(rf"c_t=f_t c_{{t-1}}+i_t\tilde c_t={f:.4f}\times{c_p:.2f}+{i:.4f}\times{cand:.4f}"
             rf"={f*c_p:.4f}+{i*cand:.4f}={c_t:.4f}")
    st.latex(rf"h_t=o_t\tanh(c_t)={o:.4f}\times\tanh({c_t:.4f})={h_t:.4f}")

    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    ax.barh(["forget f", "input i", "output o"], [f, i, o],
            color=["#C0507A", "#1D9E75", "#185FA5"], height=.55)
    ax.set_xlim(0, 1); ax.axvline(.5, color="0.75", ls=":", lw=1)
    ax.set_xlabel("gate opening  (0 = closed, 1 = fully open)", fontsize=9)
    ax.tick_params(labelsize=8)
    for j, v in enumerate([f, i, o]):
        ax.text(v + .015, j, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout(); st.pyplot(fig, width="stretch")

    st.success(
        f"**The key line is $c_t=f_t c_{{t-1}}+i_t\\tilde c_t$ — addition, not repeated "
        f"multiplication by a weight.** Differentiate it and $\\partial c_t/\\partial c_{{t-1}}"
        f"=f_t={f:.3f}$. With the forget gate **open ($f\\to1$) the gradient passes through "
        f"unchanged, forever** — the *constant error carousel*. Here $f={f:.3f}$, so over 25 steps "
        f"the gradient scales by ${f:.3f}^{{25}}={f**25:.2e}$; at $f=1$ it would be exactly **1**. "
        "The network *learns* when to hold and when to forget.", icon=":material/lock_open:")
    st.markdown(
        "**GRU** is the popular simplification: it merges the cell and hidden state and uses two "
        "gates instead of three — an **update** gate $z_t$ (one knob doing both forget and input, "
        "$h_t=(1-z_t)h_{t-1}+z_t\\tilde h_t$) and a **reset** gate. ~25 % fewer parameters, "
        "usually indistinguishable in accuracy."
    )


def _memory_task():
    st.markdown(
        "**The decisive experiment.** Show the network a value at $t=0$, then feed it *nothing* "
        "for `lag` steps, and ask what it still holds. This is the long-range dependency problem "
        "in its purest form.")
    c = st.columns(3)
    v = c[0].slider("value to remember (at t=0)", 0.5, 3.0, 1.5, 0.1, key="mt_v")
    lag = c[1].slider("lag (steps of silence)", 1, 60, 25, 1, key="mt_lag")
    wh = c[2].slider("vanilla RNN wₕ", 0.5, 1.0, 0.9, 0.01, key="mt_wh")
    fg = st.slider("LSTM forget gate f  (1.0 = hold perfectly)", 0.80, 1.00, 1.00, 0.01,
                   key="mt_f")

    hv = np.tanh(v); van = [hv]
    for _ in range(lag):
        hv = np.tanh(wh * hv); van.append(hv)
    cc = np.tanh(v); ls = [float(np.tanh(cc))]
    cells = [cc]
    for _ in range(lag):
        cc = fg * cc; ls.append(float(np.tanh(cc))); cells.append(cc)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    tt = np.arange(lag + 1)
    ax.plot(tt, np.abs(van), "-o", ms=3, lw=2, color="#C0507A", label=f"vanilla RNN (wₕ={wh})")
    ax.plot(tt, np.abs(ls), "-o", ms=3, lw=2, color="#1D9E75", label=f"LSTM cell (f={fg})")
    ax.axhline(0.1 * abs(np.tanh(v)), color="0.6", ls=":", lw=1, label="10% of the original")
    ax.set_xlabel("steps since the value was shown", fontsize=9)
    ax.set_ylabel("|signal still held|", fontsize=9)
    ax.set_title("what survives the silence", fontsize=9.5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    fig.tight_layout(); st.pyplot(fig, width="stretch")

    m = st.columns(4)
    m[0].metric("vanilla, after lag", f"{van[-1]:+.5f}")
    m[1].metric("LSTM, after lag", f"{ls[-1]:+.5f}")
    m[2].metric("LSTM cell state", f"{cells[-1]:+.5f}")
    keep = abs(ls[-1]) / max(abs(van[-1]), 1e-12)
    m[3].metric("LSTM advantage", f"{keep:,.0f}×" if np.isfinite(keep) and keep < 1e9 else "∞")
    if fg >= 0.999:
        st.success(
            f"**With the forget gate fully open the cell state is unchanged — exactly "
            f"{cells[-1]:.5f} — no matter how long the lag.** Nothing multiplies it, so nothing "
            f"decays. Meanwhile the vanilla RNN has faded to {van[-1]:.5f}. That is the whole "
            "reason LSTMs work on long sequences.", icon=":material/check_circle:")
    else:
        st.warning(
            f"With $f={fg}$ the cell leaks: it decays like $f^{{\\text{{lag}}}}="
            f"{fg**lag:.4f}$. Gating only helps if the network **learns to open the gate** — "
            "which is why LSTM forget-gate biases are often initialised **positive**, so the cell "
            "starts out remembering by default.", icon=":material/warning:")


with tab_live:
    rmode = st.radio("playground",
                     ["🔁 Hidden state", "⏱ Gradients through time (BPTT)",
                      "🚪 LSTM cell", "🧠 Memory benchmark"],
                     horizontal=True, key="rnn_mode")
    st.divider()
    if rmode.startswith("🔁"):
        _hidden_state()
    elif rmode.startswith("⏱"):
        _bptt()
    elif rmode.startswith("🚪"):
        _lstm_cell()
    else:
        _memory_task()

with tab_theory:
    st.markdown(_THEORY.replace("<DIAG/>", _DIAG_SVG), unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="rnn")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** At $w_h=0.2$ the state decays as $0.2^n$ → ≈0 in ~3–4 steps (short memory). At $w_h=0.9$ it decays as $0.9^n$ → lingers ~20+ steps. Memory length grows with $w_h$ (roughly $1/(1-w_h)$).

**2.** With $w_h>1$ the *state* still saturates at tanh's $\pm1$ limits (bounded output) — but the **gradient** isn't squashed the same way, so across steps it multiplies by $>1$ and **explodes**.

**3.** At $w_h\approx0$, $h_t\approx\tanh(w_x x_t)$: no memory at all — the RNN collapses to a plain per-step neuron on the current input.""",
        label="Live tab 1–3",
    )
    lessons.solution(
        r"""**4.** Unrolled: $h_1=\tanh(W_x x_1 + W_h h_0)$, $h_2=\tanh(W_x x_2 + W_h h_1)$, $h_3=\tanh(W_x x_3 + W_h h_2)$ — the **same $W_h$** appears at every step. Backprop-through-time multiplies by $W_h$ (and $\tanh'$) once per step, giving a factor $\approx W_h^{\,k}$: $<1$ **vanishes**, $>1$ **explodes**.

**5.** **Forget** gate: how much of the old cell state to keep. **Input** gate: how much of the new candidate to write. **Output** gate: how much of the cell to expose as the hidden state.

**6.** Streaming / online inference on very long or unbounded sequences: an RNN keeps $O(1)$ state per step, whereas full attention is $O(n^2)$ in sequence length — so for tiny-memory or endless streams, recurrence still wins.""",
        label="Concept 4–6",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
