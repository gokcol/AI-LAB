"""Transformers and a tiny next-token model (ANN module).

The capstone explains how attention is wrapped into Transformer blocks and how GPT trains
on next-token prediction. The Live model deliberately is not a Transformer: it is an MLP
over a fixed character window that isolates the objective and sampling loop.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPClassifier

import lessons

CORPUS = (
    "the cat sat on the mat and the dog sat on the log. "
    "the cat ran in the sun and the dog ran in the fog. "
    "a cat and a dog sat in the sun on a mat by the log. "
    "the dog and the cat ran in the fog to the mat and the log. "
) * 3
K = 5   # context window (characters)


@st.cache_resource(show_spinner=False)
def train_lm():
    vocab = sorted(set(CORPUS))
    c2i = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    def enc(ctx):                       # k chars -> concatenated one-hots
        v = np.zeros(K * V)
        for j, ch in enumerate(ctx):
            v[j * V + c2i[ch]] = 1.0
        return v

    X, y = [], []
    for i in range(len(CORPUS) - K):
        X.append(enc(CORPUS[i:i + K])); y.append(c2i[CORPUS[i + K]])
    model = MLPClassifier(hidden_layer_sizes=(96,), max_iter=900, random_state=0)
    model.fit(np.array(X), np.array(y))
    return model, vocab, c2i, V, enc


def _context(prompt, vocab):
    ctx = [c for c in prompt.lower() if c in set(vocab)]
    ctx = ([" "] * K + ctx)[-K:]
    return "".join(ctx)


def _distribution(model, vocab, c2i, V, enc, ctx):
    p = model.predict_proba([enc(ctx)])[0]
    probs = np.zeros(len(vocab))
    for cls, pr in zip(model.classes_, p):
        probs[cls] = pr
    return probs


def _generate(model, vocab, c2i, V, enc, prompt, n, temp, rng):
    ctx = _context(prompt, vocab)
    out = ""
    for _ in range(n):
        probs = _distribution(model, vocab, c2i, V, enc, ctx)
        q = probs ** (1.0 / temp)
        q = q / q.sum()
        ch = vocab[rng.choice(len(vocab), p=q)]
        out += ch
        ctx = (ctx + ch)[-K:]
    return out


_BLOCK_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 400 420" style="width:100%;max-width:380px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A Transformer block: input embeddings, then LayerNorm and causal self-attention with a residual add, then LayerNorm and a feed-forward MLP with a residual add; stacked N times, then a linear layer and softmax over the vocabulary giving next-token probabilities."><defs><marker id="tfa" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="398" height="418" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="58" y="66" width="284" height="244" rx="10" fill="none" stroke="#C9C8C1" stroke-dasharray="5 4"/><text x="330" y="60" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6B6A66">Transformer block  × N</text><rect x="118" y="26" width="164" height="28" rx="6" fill="#E6F1FB" stroke="#5B8FC2"/><text x="200" y="45" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#0C447C">token + position embeddings</text><rect x="150" y="78" width="100" height="22" rx="5" fill="#FFFFFF" stroke="#C9C8C1"/><text x="200" y="93" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#6B6A66">LayerNorm</text><rect x="112" y="110" width="176" height="28" rx="6" fill="#FBEAF0" stroke="#C0507A"/><text x="200" y="129" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#8A2351">causal self-attention</text><circle cx="200" cy="158" r="11" fill="#FFFFFF" stroke="#9A6A2A"/><text x="200" y="162" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">+</text><rect x="150" y="180" width="100" height="22" rx="5" fill="#FFFFFF" stroke="#C9C8C1"/><text x="200" y="195" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#6B6A66">LayerNorm</text><rect x="132" y="212" width="136" height="28" rx="6" fill="#FBEAD6" stroke="#9A6A2A"/><text x="200" y="231" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#5A3E14">feed-forward MLP</text><circle cx="200" cy="262" r="11" fill="#FFFFFF" stroke="#9A6A2A"/><text x="200" y="266" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">+</text><g stroke="#9C9B95" stroke-width="1.6" fill="none"><line x1="200" y1="54" x2="200" y2="76" marker-end="url(#tfa)"/><line x1="200" y1="100" x2="200" y2="108" marker-end="url(#tfa)"/><line x1="200" y1="138" x2="200" y2="146" marker-end="url(#tfa)"/><line x1="200" y1="169" x2="200" y2="178" marker-end="url(#tfa)"/><line x1="200" y1="202" x2="200" y2="210" marker-end="url(#tfa)"/><line x1="200" y1="240" x2="200" y2="250" marker-end="url(#tfa)"/><path d="M74,70 V158 H187" marker-end="url(#tfa)"/><path d="M74,172 V262 H187" marker-end="url(#tfa)"/><line x1="200" y1="273" x2="200" y2="326" marker-end="url(#tfa)"/><line x1="200" y1="354" x2="200" y2="372" marker-end="url(#tfa)"/></g><text x="92" y="120" font-family="sans-serif" font-size="9" fill="#9C9B95" transform="rotate(-90 92 120)">residual</text><rect x="110" y="328" width="180" height="26" rx="6" fill="#FFFFFF" stroke="#5B8FC2"/><text x="200" y="345" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">Linear → softmax (vocab)</text><rect x="120" y="374" width="160" height="28" rx="6" fill="#D7EFE5" stroke="#1D9E75"/><text x="200" y="392" text-anchor="middle" font-family="sans-serif" font-size="11.5" fill="#0E5E45">P(next token)</text></svg></div>'''

_MASK_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 300 240" style="width:100%;max-width:300px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A causal attention mask: a 5 by 5 grid where each row (query position) can attend only to columns at or before it — the lower-triangular cells are filled, the upper ones blocked."><rect x="1" y="1" width="298" height="238" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="150" y="24" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">causal mask — attend to ≤ i only</text>''' + "".join(
    f'<rect x="{60+j*36}" y="{36+i*36}" width="34" height="34" '
    f'fill="{"#CFE3F5" if j<=i else "#EFEEE8"}" stroke="#FFFFFF"/>'
    + (f'<text x="{77+j*36}" y="{58+i*36}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0C447C">✓</text>' if j <= i else "")
    for i in range(5) for j in range(5)
) + '''<text x="40" y="130" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66" transform="rotate(-90 40 130)">query i →</text><text x="150" y="232" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66">key j (position) →</text></svg></div>'''


_THEORY = r"""
## 1. The whole picture

A GPT is one pipeline: **tokens → embeddings (+ positions) → N Transformer blocks → a
softmax over the vocabulary**. Each block refines every token's vector by (a) letting it
gather context with attention and (b) processing it with a small MLP. After $N$ blocks, each
position's vector is used to predict the **next** token.

<BLOCK/>

## 2. The Transformer block

Attention (previous page) is the *mixer* — it moves information **between** tokens. The block
pairs it with a per-token **MLP** that does the *processing* on each token independently,
and wraps both in two stabilizers:

- **Residual connections** — each sub-layer computes $x \leftarrow x + f(x)$, **adding** to
  its input instead of replacing it. That gives gradients a clean "highway" straight back
  through many layers, so the vanishing-gradient problem (Backprop §11, ANN §6) doesn't kill
  deep stacks — and it lets each layer learn a small *refinement* rather than a whole new
  representation.
- **LayerNorm** — re-centers and re-scales each token's vector (to mean 0 / unit variance,
  then a learned scale & shift) so activations stay in a healthy range as depth grows.

So a block is **`x = x + attn(norm(x))`** then **`x = x + mlp(norm(x))`**. Stack $N$ of them
→ the body of a GPT. (Roughly half the parameters live in those MLPs.)

## 3. Causal masking — only look back

A language model predicts the **next** token, so when processing position $i$ it must not
peek at the future. **Causal (masked) self-attention** sets the scores for any position
$j>i$ to $-\infty$ *before* the softmax, so they get **0 weight** — a lower-triangular
pattern:

<MASK/>

This mask is the *only* change from the Attention page's self-attention — and it's exactly
what lets the model train on **every position at once** (§5) while never cheating.

## 4. The output head — next-token softmax

After the last block (and a final LayerNorm), each token's vector passes through a **linear
layer → softmax over the whole vocabulary**, giving $P(\text{next token}\mid\text{everything so far})$ at that
position. The linear layer has **one row per vocabulary item**; the dot
product of a token's vector with each row scores how likely that item comes next (Math X1,
once more).

## 5. Training — next-token prediction

One objective: **make the true next token likely** — minimize the **cross-entropy** between
the predicted distribution and the actual next token (M2 / X5), averaged over the corpus.
Two things make it scale:

- **Shifted targets (teacher forcing)** — the target at position $i$ is just the input token
  at position $i{+}1$. No human labels: the text **is its own supervision**
  (*self-supervised*).
- **All positions in parallel** — thanks to the causal mask, one forward pass yields a
  prediction at *every* position, and the loss is averaged over all of them.

That's the *entire* learning signal — predict the next token, billions of times — yet from
it the model absorbs grammar, facts, style, and reasoning patterns.

## 6. Generation = autoregressive sampling

To produce text: feed the prompt, read the next-token distribution at the **last** position,
**pick** a token, append it, and repeat — feeding the model its own output
(*autoregressive*). How you pick matters:

- **Greedy** — always the most likely token: safe but repetitive.
- **Temperature** $T$ — divide the logits by $T$ before softmax: $T<1$ sharpens (safer),
  $T>1$ flattens (more random/creative) — the same knob as the Attention page (X5).
- **Top-k / top-p** — sample only from the $k$ most likely tokens (or the smallest set with
  probability $\ge p$), cutting off the long tail of nonsense.

The Live tab uses temperature plus a small top-k.

## 7. Why it scales — and the honest caveat

Make the blocks wider, stack more, and train on more text, and this same recipe keeps
improving in a strikingly predictable way (**scaling laws**), with new capabilities
appearing along the way. A production LLM differs from this page only in **data, parameter
count, and compute** — plus engineering (a better tokenizer, normalization and positional
schemes, and a KV-cache for fast generation).

> The Live demo is a fixed-window neural LM standing in for the stacked-attention version,
> but the **output head (next-token softmax)**, the **objective (cross-entropy)**, and
> **generation (temperature sampling)** are exactly a GPT's. For the *real* stacked-attention
> model trained by backprop, run experiment **e21 (nanoGPT)**.

## 8. That's a GPT

Stack masked multi-head self-attention blocks, add token + position embeddings and a
next-token head, train by cross-entropy, generate by sampling. The whole lab built to here:
**neuron → MLP → backprop → optimizers → attention → Transformer → a language model.**
*(Roadmap Tier 5; experiment e21 trains the real thing.)*

## 9. What you just built is a *generative* model

Worth naming, because the word is everywhere and this is it. A **discriminative** model
learns $p(y\mid x)$ — given an input, which label? That is every classifier in the ML
track. A **generative** model learns $p(x)$ itself: the distribution the data came from,
so you can *draw new samples* from it.

A language model looks like it only predicts one token, so where does $p(x)$ come from?
The **chain rule of probability** — and note this is an identity, not an approximation:

$$ p(x_1,\dots ,x_T)\;=\;\prod_{t=1}^{T} p(x_t \mid x_{<t}) $$

Every joint distribution over a sequence factorises this way, exactly. So a model that is
good at *one* conditional $p(x_t\mid x_{<t})$, applied at every position, **is** a model of
the whole sequence. Concretely, for `the cat`:

$$ p(\texttt{the cat}) = p(\texttt{t})\,p(\texttt{h}\mid \texttt{t})\,p(\texttt{e}\mid \texttt{th})\cdots p(\texttt{t}\mid \texttt{the ca}) $$

Which means the training objective from §5 was never merely "guess the next character".
Minimising cross-entropy summed over positions

$$ \mathcal L = -\sum_{t} \log p(x_t\mid x_{<t}) \;=\; -\log p(x_1,\dots ,x_T) $$

is **maximum likelihood on entire sequences**. Next-token prediction is a generative
objective wearing a classifier's clothes — and sampling (§6) is drawing from the
distribution it learned.

Two consequences worth seeing:

- **The causal mask is the enabling structure, not a handicap.** It is what makes each
  conditional depend only on the past, so the factorisation is valid — and because
  position $t$ cannot see the future, all $T$ conditionals can be trained *in one forward
  pass* instead of $T$ of them. The mask is why this is cheap.
- **A perplexity is a likelihood.** $\exp(\mathcal L/T)$ is the standard LM score, so the
  benchmark everyone quotes is just "how probable does the model think real text is".

This is the **autoregressive** branch of generative modelling: factorise, then predict one
piece at a time. It is not the only branch — VAEs, GANs and diffusion models attack $p(x)$
in completely different ways, and diffusion, not autoregression, is what generates images.
The **Generative family** page lays the four side by side.

## 10. The block, written out

Every modern block is two sub-layers, each wrapped in a residual connection with
normalization applied **before** it (*pre-norm*):

$$ x \leftarrow x + \mathrm{MHA}\big(\mathrm{Norm}(x)\big), \qquad x \leftarrow x + \mathrm{FFN}\big(\mathrm{Norm}(x)\big) $$

The two halves have complementary jobs: **attention mixes information _between_ positions**,
while the **FFN processes each position independently** — the "thinking" step applied
token-wise. The residual `x + ...` is what lets gradients reach layer 1 of a 96-layer stack
(the same trick as ResNet).

The **FFN** expands and contracts, conventionally by 4×:
$$ \mathrm{FFN}(x)=W_2\,\varphi(W_1x),\qquad W_1\in\mathbb R^{4d\times d},\; W_2\in\mathbb R^{d\times 4d} $$
with $\varphi$ = GELU (GPT-2/3) or a gated **SwiGLU** (Llama, PaLM). Note the FFN holds
**twice** the parameters of the attention it follows.

## 11. Counting the parameters

Per block: $4d^2$ (attention) $+\;8d^2$ (FFN) $=\;\mathbf{12d^2}$. So

$$ N \;\approx\; \underbrace{12\,d^2 L}_{\text{blocks}} \;+\; \underbrace{V d}_{\text{embeddings}} $$

**Worked, for GPT-2 small** ($d=768$, $L=12$, $V=50\,257$, ctx 1024):
$12\cdot768^2\cdot12 = 84.9\text{M}$, embeddings $50\,257\cdot768=38.6\text{M}$, positions
$0.8\text{M}$ → **124.3 M**, exactly the published 124 M. The same formula gives GPT-3
($d=12\,288$, $L=96$): $12d^2L \approx 174\text{B}$ ≈ the published 175 B.

Two consequences: parameters grow with $d^2$, so **width is expensive**; and in small models
the *embedding table* is a large share of the total (31 % here), which is why many models
**tie** the input embedding and output-head weights.

## 12. Training: one objective, a lot of tokens

The loss is plain **cross-entropy** on the next token, averaged over every position — one
sequence of length $n$ yields $n$ training signals at once, which is why self-supervised
pretraining is so efficient. **Teacher forcing** feeds the true prefix during training
(so all positions can be computed in parallel), while the **causal mask** prevents cheating.

**How much data?** The Chinchilla result: for a compute budget, parameters and tokens should
scale *together* — roughly **20 tokens per parameter**. A 7 B model wants ~140 B tokens; a
70 B model ~1.4 T. Earlier models (GPT-3) were badly under-trained by this measure, which is
why later 7 B models beat them.

## 13. Inference: the KV cache, and why generation is slow

Generating token $t{+}1$ re-attends over all previous tokens — but their keys and values were
already computed, so they are **cached** rather than recomputed. That turns per-token cost
from $O(n^2)$ to $O(n)$, at the price of memory:

$$ \text{KV bytes} = 2\,\cdot\,L\,\cdot\,n_{\text{heads}}\,\cdot\,d_h\,\cdot\,n\,\cdot\,\text{batch}\,\cdot\,\text{bytes} $$

For a 32-layer, 32-head, $d_h{=}128$ model at 32 k context with batch 2, that is ≈ **17 GB**
in fp16 — often more than the weights. This is why generation is **memory-bandwidth bound**
rather than compute bound, why long contexts are expensive to *serve*, and why **GQA** (fewer
KV heads) was adopted so quickly.

Note the asymmetry: **training** processes all positions in parallel, but **generation** is
inherently sequential — one token at a time, forever.

## 14. Three families from one block

| family | attention | trained to | good at |
|---|---|---|---|
| **Decoder-only** (GPT, Llama, Claude) | causal | predict the next token | generation — and, it turns out, everything |
| **Encoder-only** (BERT) | bidirectional | fill in masked tokens | understanding: classification, retrieval |
| **Encoder–decoder** (T5, original) | both + cross-attention | map sequence → sequence | translation, summarization |

The field consolidated on **decoder-only** because one simple objective scales, and because
generation subsumes the other tasks when the model is large enough (just ask it in words).

## 15. Mixture of Experts — more parameters, same compute

The FFN is where most parameters live, so **MoE** replaces it with $E$ expert FFNs plus a
small router that sends each token to only the top-1 or top-2 experts. Total parameters rise
enormously while **compute per token stays roughly flat** — a way to buy capacity without
buying FLOPs. Used in Mixtral, and widely believed to be in the frontier closed models.

## 16. What actually made it work

Worth separating the ideas from the engineering:

1. **Attention** — direct, $O(1)$-path communication between any two positions.
2. **Residuals + normalization** — makes 100-layer depth trainable at all.
3. **Parallelism over positions** — the reason it beat RNNs on GPUs; the architecture was
   designed to *fit the hardware*.
4. **One self-supervised objective** — next-token prediction needs no labels, so the training
   set is "the internet".
5. **Scale** — the same block, made wider and deeper on more tokens, kept improving
   predictably (scaling laws).

Nothing on that list is a new kind of *neuron*. It is the same weighted sum and nonlinearity
from Level 1, wired so that information can move freely and the whole thing fits a GPU.
"""

_QUIZ = [
    lessons.Question(
        "A Transformer block is, in essence:",
        ["just one big matrix multiply", "self-attention + a per-token MLP, each wrapped with a residual add and LayerNorm",
         "a convolution followed by pooling", "k-means over tokens"], 1,
        "Attention mixes tokens, the MLP processes each one; residuals + LayerNorm make deep stacks trainable."),
    lessons.Question(
        "What do residual connections buy a deep Transformer?",
        ["fewer parameters", "a gradient 'highway' that keeps deep stacks trainable (avoids vanishing)",
         "causal masking", "a larger vocabulary"], 1,
        "Adding to the input (instead of replacing it) lets gradients flow back through many layers — ANN §6."),
    lessons.Question(
        "Why the causal (triangular) mask?",
        ["to save memory", "so position i can't attend to future positions when predicting the next token",
         "to make attention symmetric", "to remove the softmax"], 1,
        "A next-token predictor must only use the past; the mask blocks attention to j > i."),
    lessons.Question(
        "How is a GPT trained, fundamentally?",
        ["clustering tokens", "predict the next token — minimize cross-entropy against the true next token, over a huge corpus",
         "matching images to labels", "k-fold cross-validation"], 1,
        "One self-supervised objective — next-token prediction (softmax + cross-entropy) — does it all."),
]

_TASKS = r"""
### In the Live tab
1. Type a prompt like `the ` and read the **next-character distribution** — which letters
   does the model expect, and do they match the corpus?
2. **Generate** at low **temperature** (≈0.4) vs high (≈1.3): low is repetitive/safe, high
   is wilder. Relate this to the softmax temperature from the Attention page (X5).
3. Find a temperature that produces the most "word-like" text. Why is there a sweet spot?

### Concept
4. Explain why a residual connection helps train a 50-layer Transformer (tie to ANN §6).
5. Sketch the causal mask for a length-4 sequence; which entries are zero and why?

### Bridge
6. Connect the dots end to end: **dot product (X1)** → **softmax (X5)** → **attention** →
   **block (residual + LayerNorm + MLP)** → **next-token cross-entropy (M2/X5)** = a GPT.
"""

_REFS = r"""
- Vaswani et al. (2017) — *Attention Is All You Need* (the Transformer).
- Radford et al. — *GPT / GPT-2* (decoder-only, next-token LM).
- Karpathy — *Let's build GPT* + **nanoGPT** (a real tiny GPT in code).
- Jay Alammar — *The Illustrated GPT-2*.
- In this lab: **Attention**, **MLP**, **Backprop**, Math **X5** (softmax/cross-entropy),
  roadmap **Tier 5**.
"""


st.title("Transformers & a tiny next-token model")
st.caption("Learn the Transformer architecture, then isolate its training objective in a small "
           "fixed-window character model you can prompt and sample.")

lessons.predict(
    'This tiny model trains on **one** objective. What is it — and how does the *same* model then **generate** text?',
    "Predict the **next token** (softmax + cross-entropy over the vocabulary). Generation is sampling that distribution, appending the choice, and repeating. A GPT uses the same objective and loop, but replaces this live demo's fixed-window MLP with stacked causal Transformer blocks.",
)

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["⌨ Generate", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("A tiny **character-level** language model trained on a few sentences — it "
                "predicts the **next character** from the last few. Real GPTs replace the "
                "fixed window with stacked attention, but the next-token softmax + sampling "
                "are the same.")
    with st.expander("training corpus (what it learned from)"):
        st.code(CORPUS[:220] + " …", language=None)

    model, vocab, c2i, V, enc = train_lm()

    prompt = st.text_input("prompt", value="the cat ", key="tf_prompt")
    cc = st.columns(2)
    temp = cc[0].slider("temperature", 0.3, 1.5, 0.7, 0.1, key="tf_temp",
                        help="low = safe/repetitive · high = random/creative")
    nchars = cc[1].slider("characters to generate", 20, 160, 80, key="tf_n")

    ctx = _context(prompt, vocab)
    probs = _distribution(model, vocab, c2i, V, enc, ctx)
    order = np.argsort(probs)[::-1][:12]
    labels = [("␣" if vocab[i] == " " else vocab[i]) for i in order]
    st.caption(f"Next-character probabilities after context “{ctx.replace(' ', '␣')}”")
    st.bar_chart(pd.DataFrame({"P(next)": probs[order]}, index=labels), height=220)

    if st.button("Generate ▶", key="tf_gen"):
        rng = np.random.default_rng()
        text = _generate(model, vocab, c2i, V, enc, prompt, nchars, temp, rng)
        st.markdown(f"**{prompt}**{text}")
    st.info("This live model is an **MLP over five one-hot characters**, not a Transformer. "
            "It demonstrates next-token cross-entropy and autoregressive sampling only. A GPT "
            "uses the same objective and loop with stacked causal attention blocks.",
            icon=":material/smart_toy:")
    st.caption("⚡ The **real, GPU-trained** version is experiment **e21** — a PyTorch "
               "Transformer (causal multi-head attention + AdamW) at "
               "`experiments/tier4_attention/e21_nanogpt`. Run it from the **Experiments** "
               "page or the CLI (`--download --iters 3000` for Shakespeare).")

with tab_theory:
    st.markdown(_THEORY.replace("<BLOCK/>", _BLOCK_SVG).replace("<MASK/>", _MASK_SVG),
                unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="transformer")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** After `the ` the model expects the characters that most often follow it in the corpus (a space, then common letters) — its distribution mirrors the training text's statistics.

**2.** Low temperature (≈0.4) → safe, repetitive text (it keeps picking high-probability characters); high (≈1.3) → wilder, noisier. It's the same softmax temperature as the Attention page (X5), applied to the next-token distribution.

**3.** There's a sweet spot because too-low temperature loops on the single most likely character, while too-high is near-random gibberish; a middle value balances coherence and variety → the most word-like output.""",
        label="Live tab 1–3",
    )
    lessons.solution(
        r"""**4.** A residual connection outputs $x + f(x)$, so gradients flow straight through the identity path to earlier layers — they don't vanish through 50 stacked blocks. That's the vanishing-gradient fix from ANN §6.

**5.** The causal mask is **lower-triangular**: for a length-4 sequence, entry $(i,j)$ is allowed when $j\le i$ and set to $-\infty$ (→ 0 weight after softmax) when $j>i$, so a token can never attend to a **future** token.""",
        label="Concept 4–5",
    )
    lessons.solution(
        r"""**6.** End to end: **dot product (X1)** scores similarity → **softmax (X5)** makes weights → **attention** mixes values → a **block** wraps it with a residual + LayerNorm + MLP → stack the blocks and train on **next-token cross-entropy (M2/X5)**. That stack *is* a GPT.""",
        label="Bridge 6",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
