"""Attention — the engine of Transformers & LLMs (ANN module).

Self-attention lets each token gather information from every other token by **dot-product
similarity → softmax → weighted sum** — exactly the dot product (Math X1) and softmax
(Math X5) you already know, composed. The Live tab runs scaled dot-product self-attention
on a tiny sentence with hand-set embeddings, so the attention pattern is interpretable.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lessons

TOKENS = ["cat", "dog", "sat", "ran", "."]
# Hand-set embeddings so the pattern is readable: cat~dog (animals), sat~ran (verbs),
# "." stands alone. In a real model these are *learned*.
E = np.array([
    [1.0, 0.2, 0.0],   # cat
    [0.9, 0.3, 0.0],   # dog
    [0.1, 1.0, 0.2],   # sat
    [0.0, 0.9, 0.3],   # ran
    [0.2, 0.2, 1.0],   # .
])
D = E.shape[1]


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _attention(temp=1.0):
    scores = (E @ E.T) / np.sqrt(D)
    W = _softmax(scores / temp)
    out = W @ E
    return scores, W, out


_FLOW_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 700 195" style="width:100%;max-width:700px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Scaled dot-product attention: queries and keys form a score matrix Q times K-transpose, divided by the square root of d, softmaxed into attention weights, then multiplied by the values V to give the output context vectors."><defs><marker id="atf" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="698" height="193" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.5"><rect x="22" y="36" width="64" height="30" rx="6"/><rect x="22" y="82" width="64" height="30" rx="6"/><rect x="22" y="146" width="64" height="30" rx="6"/></g><g font-family="sans-serif" font-size="13" text-anchor="middle" fill="#0C447C"><text x="54" y="56">Q</text><text x="54" y="102">K</text><text x="54" y="166">V</text></g><g font-family="sans-serif" font-size="9" text-anchor="middle" fill="#6B6A66"><text x="54" y="76">query</text><text x="54" y="122">key</text><text x="54" y="186">value</text></g><g fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.5"><rect x="150" y="70" width="104" height="40" rx="8"/><rect x="284" y="70" width="74" height="40" rx="8"/></g><rect x="388" y="70" width="104" height="40" rx="8" fill="#FBEAF0" stroke="#C0507A" stroke-width="1.5"/><rect x="522" y="70" width="128" height="40" rx="8" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.5"/><g font-family="sans-serif" font-size="12" text-anchor="middle" fill="#33312E"><text x="202" y="94">Q·Kᵀ scores</text><text x="321" y="94">÷ √d</text><text x="440" y="94" fill="#8A2351">softmax</text><text x="586" y="89">weighted</text><text x="586" y="103">sum × V</text></g><g stroke="#9C9B95" stroke-width="1.7" fill="none"><line x1="88" y1="52" x2="148" y2="82" marker-end="url(#atf)"/><line x1="88" y1="96" x2="148" y2="96" marker-end="url(#atf)"/><line x1="254" y1="90" x2="282" y2="90" marker-end="url(#atf)"/><line x1="358" y1="90" x2="386" y2="90" marker-end="url(#atf)"/><line x1="492" y1="90" x2="520" y2="90" marker-end="url(#atf)"/><line x1="88" y1="160" x2="560" y2="112" marker-end="url(#atf)"/><line x1="650" y1="90" x2="688" y2="90" marker-end="url(#atf)"/></g><text x="440" y="128" text-anchor="middle" font-family="sans-serif" font-size="9.5" fill="#8A2351">attention weights</text><text x="694" y="86" text-anchor="end" font-family="sans-serif" font-size="10" fill="#0E5E45">context</text></svg></div>'''


_THEORY = r"""
## 1. Why attention

An MLP mixes its inputs with **fixed** weights — it treats position 3 the same way every
time. An RNN threads a sequence through one small running state, so distant tokens blur and
it must process **step by step** (slow). **Attention** lets each token **dynamically pull
information from any other token** — near or far, all in parallel — choosing *what to look
at* based on **content**, not position. That one idea removed the sequence bottleneck and
made **Transformers** (and modern **LLMs**) possible.

## 2. Tokens are vectors

First, text is split into **tokens** (here: characters; real LLMs use sub-word pieces), and
each token becomes a learned **embedding** vector. A length-$n$ sequence is then a matrix
$X$ of shape $n \times d$ (sequence length × model width). Everything below is operations on
these vectors — and "similarity" between them *is* the dot product (Math X1).

## 3. Query, Key, Value

From each token's vector the model makes three new vectors via **learned linear
projections** $W_Q,W_K,W_V$:
$$ Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V. $$
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what I'll hand over if you attend to me."

A token's query is matched against **every** token's key to decide how much of each token's
value to take. (Q/K/V are three learned "views" of the same token — that's why the
projections are separate, trainable matrices.)

## 4. Scaled dot-product attention

Three steps — **dot product → softmax → weighted sum**:
$$ \text{Attention}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)\,V. $$
1. **Scores** $QK^\top$ — an $n\times n$ matrix; entry $(i,j)$ is query $i$ · key $j$ = how
   well token $i$ matches token $j$ (Math X1).
2. **Scale & softmax** — divide by $\sqrt{d_k}$ (§6), then softmax **each row** (Math X5), so
   every token's scores become a probability distribution over all tokens — its **attention
   weights**.
3. **Weighted sum** — multiply by $V$: token $i$'s output is the attention-weighted blend of
   *every* token's value — a new, **context-aware** representation.

<FLOW/>

## 5. A tiny worked example

Suppose token $i$'s query yields raw scores $[2,\,0,\,1]$ against three keys (take
$d_k=1$, no scaling). Softmax → $[0.67,\,0.09,\,0.24]$. The output is
$0.67\,v_1 + 0.09\,v_2 + 0.24\,v_3$ — mostly token 1's value, a dash of token 3's. Nudge the
query to align with key 2 and the blend shifts toward $v_2$. That, per token and in parallel,
is the whole mechanism — the Live tab shows the full $n\times n$ weight matrix.

## 6. Why divide by √d

Dot products of $d_k$-dimensional vectors grow like $\sqrt{d_k}$. Left unscaled, large scores
shove softmax into its **saturated** corners (one weight ≈ 1, the rest ≈ 0) where its
gradient is almost zero — the vanishing problem (ANN §6). Dividing by $\sqrt{d_k}$ keeps
scores in a sane range so attention stays smooth and trainable.

## 7. Self-attention & multi-head

- **Self-attention** — Q, K, V all come from the *same* sequence, so every token attends to
  the others (what the Live tab shows). *(When they come from two sequences it's
  cross-attention — e.g. translation.)*
- **Multi-head** — split the $d$ dimensions into $h$ smaller **heads**, run attention in each
  independently, then concatenate and project. Each head can specialize — one tracks local
  context, another long-range coreference, another syntax — capturing several relationship
  types at once.

## 8. Position has to be added back

Attention is a weighted sum over a **set**, so on its own it has **no sense of order**
("dog bites man" ≈ "man bites dog" to raw attention). Transformers therefore add a
**positional encoding** to each token embedding — a fixed sinusoid or a learned
per-position vector — telling the model *where* each token sits. (The Tiny GPT page uses
learned position embeddings.)

## 9. The cost

The score matrix is $n\times n$, so self-attention is **$O(n^2)$** in time and memory —
quadratic in sequence length. That's why context windows are bounded and why lots of LLM
research chases cheaper/longer attention. The upside: it's **one big matrix multiply**, so
it parallelizes on GPUs — unlike an RNN's step-by-step loop, which is why Transformers
train so much faster.

## 10. It's all things you already know

Attention adds **no new math**: a **dot product** for similarity (Math X1), **softmax** for
a distribution (Math X5), a **weighted sum**, and learned projections (matrix multiplies,
X1). The genius is the *composition* — at every layer the data itself decides which tokens
inform which. Add a causal mask + an MLP into a **block**, repeat, cap with a next-token
softmax head, and you have a **GPT** (Tiny GPT page). Everything in this lab leads here:
neuron → MLP → backprop → attention → Transformer → LLM. *(Roadmap Tier 4–5.)*

## 11. The shapes and the parameter count

For a model width $d$ and sequence length $n$, one attention layer holds four square
matrices — $W_Q, W_K, W_V$ and the output projection $W_O$:

$$ X\in\mathbb R^{n\times d},\quad Q=XW_Q,\; K=XW_K,\; V=XW_V \in \mathbb R^{n\times d},\qquad \text{params} = 4d^2 $$

The attention matrix $QK^\top/\sqrt{d}$ is $n\times n$ — one score for every **pair** of
positions — and after softmax it multiplies $V$ back to $n\times d$. Note what is *not* in
that list: **no parameter depends on $n$**, which is why the same weights work at any
sequence length.

## 12. Multi-head attention, precisely

One attention pattern per layer is limiting — a token may need to track its subject *and* its
tense *and* its position at once. So we split the width into $h$ **heads** of size
$d_h = d/h$, run attention independently in each, then concatenate and project:

$$ \mathrm{head}_i=\mathrm{Attn}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}),\qquad \mathrm{MHA}(X)=\big[\mathrm{head}_1;\dots;\mathrm{head}_h\big]W_O $$

Crucially this is **free**: $h$ heads of size $d/h$ cost the same as one head of size $d$.
GPT-2 small uses $d=768$ with $h=12$, so $d_h=64$. Interpretability work finds heads
specialise — *previous-token* heads, *syntactic* heads, and famously **induction heads**
that implement "I saw `A B` earlier, and here is `A` again, so predict `B`" — the mechanism
behind much of in-context learning.

## 13. Positions: sinusoidal → learned → RoPE

Attention is **permutation-equivariant** (§8): shuffle the tokens and the outputs shuffle with
them. Order must be injected explicitly, and how has changed over time:

| scheme | idea | used by |
|---|---|---|
| **Sinusoidal** | fixed sin/cos of different frequencies, added to the embedding | original Transformer |
| **Learned absolute** | a trainable vector per position | GPT-2, BERT |
| **RoPE** (rotary) | *rotate* Q and K by an angle proportional to position, so the dot product depends on **relative** distance | **Llama, GPT-NeoX, most modern LLMs** |
| **ALiBi** | add a distance-proportional penalty to the scores | BLOOM, MPT |

The move to **relative** schemes matters because absolute embeddings do not extrapolate:
a model with 1 024 learned position vectors has literally nothing to use at position 1 025.
RoPE degrades gracefully and can be *interpolated* to extend context after training — which
is how models get long-context versions without retraining from scratch.

## 14. The cost, and how it is fought

Attention is $O(n^2 d)$ in time and $O(n^2)$ in memory: at $n=1024$ that is ~1 M scores; at
$n=131\,072$ it is **17 billion**. Four responses, all in production use:

- **FlashAttention** — an *exact* algorithm that never materialises the $n\times n$ matrix,
  tiling the computation to stay in fast on-chip SRAM. Same maths, several times faster.
- **GQA / MQA** — let several query heads *share* one key/value head. Barely changes quality
  and shrinks the KV cache severalfold (Llama 2 70B onward).
- **Sliding-window / sparse attention** — each token attends only to a local window (Mistral),
  making cost linear in $n$.
- **Linear-attention & state-space models** (Mamba) — drop softmax attention for an $O(n)$
  recurrence, at some cost in recall.

## 15. Self- vs cross-attention

Same equations, different sources. In **self-attention** $Q,K,V$ all come from the same
sequence — that is what a GPT block uses. In **cross-attention** the queries come from one
sequence and the keys/values from another: a translation decoder queries the encoder's
representation of the source sentence, and a text-to-image model has image queries attending
to text keys. The mechanism is identical; only *where you take the vectors from* changes.

## 16. The mental model worth keeping

Attention is a **soft, differentiable dictionary lookup**. A hard lookup takes a key, finds
the exact match, returns its value. Attention takes a query, scores *every* key by similarity,
softmaxes the scores into weights, and returns the **weighted blend of all values**. Because
every step is differentiable, the network can *learn what to look up* — and that, rather than
any new kind of neuron, is the whole innovation.
"""

_QUIZ = [
    lessons.Question(
        "In attention, what do the Query and Key produce together?",
        ["the final output directly", "a similarity score (dot product) saying how much one token should attend to another",
         "the loss", "the embedding"], 1,
        "score(i,j) = qᵢ·kⱼ/√d — a dot-product similarity; softmax turns a token's scores into attention weights."),
    lessons.Question(
        "After softmax, the output for a token is…",
        ["the largest value vector", "a weighted sum (blend) of all tokens' Value vectors",
         "the query vector unchanged", "a random token"], 1,
        "output_i = Σ_j weight(i,j)·v_j — a context-aware blend of values."),
    lessons.Question(
        "Why divide the scores by √d?",
        ["to save memory", "to keep dot products from getting huge and saturating softmax (tiny gradients)",
         "to make them integers", "it's optional and cosmetic"], 1,
        "Scores grow like √d; scaling keeps softmax out of its saturated corners so gradients stay useful."),
    lessons.Question(
        "Attention is built from which familiar pieces?",
        ["convolution + pooling", "dot product (similarity, X1) + softmax (distribution, X5) + weighted sum",
         "eigenvectors + SVD", "k-means + PCA"], 1,
        "No new math — just those three composed so the data chooses what to attend to."),
]

_TASKS = r"""
### In the Live tab
1. Pick **cat** as the query — which tokens get the most weight, and why? Do the same for
   **sat**. (Animals attend to animals, verbs to verbs.)
2. Lower the **temperature** toward 0 — the attention sharpens toward one token (hard
   selection); raise it — attention flattens toward uniform. Relate this to softmax (X5).
3. Read the **heatmap**: which 2×2 blocks light up, and what relationship do they encode?

### Pencil & paper
4. With $q=(1,0)$ and keys $k_1=(1,0),\,k_2=(0,1),\,k_3=(-1,0)$, rank the attention weights
   before softmax. Which value gets blended in most?
5. Explain in one sentence why attention handles long-range dependencies better than an RNN.

### Bridge
6. Connect: dot product = **Math X1**, softmax = **Math X5**, the next-token loss =
   **M2 / X5** cross-entropy. Attention is these composed.
"""

_REFS = r"""
- Vaswani et al. (2017) — **"Attention Is All You Need"** (the Transformer paper).
- Jay Alammar — **The Illustrated Transformer** (the best visual walk-through).
- Karpathy — **"Let's build GPT"** / nanoGPT (attention in code, from scratch).
- 3Blue1Brown — *Attention in transformers* (visual intuition).
- In this lab: Math **X1** (dot product), Math **X5** (softmax / cross-entropy), the
  **MLP** + **Backprop** pages, and roadmap **Tier 3–5**.
"""


st.title("Attention — the engine of Transformers & LLMs")
st.caption("Each token gathers info from the others by dot-product similarity → softmax → "
           "weighted sum. Self-attention runs live below on a tiny sentence.")

lessons.predict(
    'A query token attends most to the token whose key it is most ___ to. Which operation *measures* that, and which turns the scores into weights?',
    'Most **similar** to — measured by the **dot product** (Math X1). **Softmax** (X5) turns the raw scores into weights that sum to 1, and the output is the weighted sum of values. Raise the temperature → weights spread out; lower it → attention sharpens onto one token.',
)

tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔎 Live attention", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_live:
    st.markdown("Self-attention over the tokens **" + " · ".join(f"`{t}`" for t in TOKENS) +
                "** (hand-set embeddings: *cat≈dog*, *sat≈ran*, *.* alone — in a real model "
                "these are learned). Q = K = V = the embeddings here.")
    cc = st.columns(2)
    temp = cc[0].slider("temperature (softmax sharpness)", 0.2, 2.0, 1.0, 0.1, key="att_T",
                        help="lower = sharper / harder attention, higher = flatter")
    qi = cc[1].selectbox("inspect query token", TOKENS, index=0, key="att_q")

    scores, W, out = _attention(temp)
    qidx = TOKENS.index(qi)

    left, right = st.columns(2)
    with left:
        st.caption("Attention weights — row = query, column = attended token")
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        im = ax.imshow(W, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(TOKENS))); ax.set_xticklabels(TOKENS)
        ax.set_yticks(range(len(TOKENS))); ax.set_yticklabels(TOKENS)
        ax.set_xlabel("attends to →"); ax.set_ylabel("query token")
        for i in range(len(TOKENS)):
            for j in range(len(TOKENS)):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        color="white" if W[i, j] > 0.5 else "#333", fontsize=8)
        st.pyplot(fig, width="stretch")
    with right:
        st.caption(f"How **{qi}** distributes its attention")
        st.bar_chart(pd.DataFrame({"weight": W[qidx]}, index=TOKENS), height=300)

    df = pd.DataFrame({
        "token": TOKENS,
        "q·k (raw)": np.round(E[qidx] @ E.T, 2),
        "scaled ÷√d": np.round((E[qidx] @ E.T) / np.sqrt(D), 2),
        "attention": np.round(W[qidx], 3),
    })
    st.dataframe(df, hide_index=True, width="content")
    top = TOKENS[int(np.argmax(W[qidx]))]
    st.success(f"**{qi}** attends most to **{top}** "
               f"({W[qidx].max():.0%}). Its output vector is the attention-weighted blend of "
               "all the Value vectors — a context-aware version of the token.",
               icon=":material/center_focus_strong:")
    st.info("Same three pieces you already know: **dot product** (Math X1) for similarity, "
            "**softmax** (Math X5) for the weights, then a **weighted sum**.",
            icon=":material/bolt:")

with tab_theory:
    st.markdown(_THEORY.replace("<FLOW/>", _FLOW_SVG), unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="attention")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** **cat** attends most to related tokens (other animals / itself); **sat** to verbs — because their learned query and key vectors have a high dot product (similarity).

**2.** Low temperature → softmax **sharpens** onto the single top-scoring token (hard selection); high temperature → **flattens** toward uniform. It's the same softmax-temperature knob you meet again in decoding (X5).

**3.** The bright 2×2 blocks are groups of related tokens attending to one another (animal↔animal, verb↔verb) — attention has grouped the sentence by relationship.""",
        label="Live tab 1–3",
    )
    lessons.solution(
        r"""**4.** Scores $q\cdot k$: $q\cdot k_1=1$, $q\cdot k_2=0$, $q\cdot k_3=-1$. Ranked $k_1 > k_2 > k_3$, so value $v_1$ is blended in most (and $v_3$ least).

**5.** Attention connects any two positions in **one** step (a direct dot product), whereas an RNN must carry information through many sequential steps where it decays — so attention captures long-range dependencies far better.""",
        label="Pencil & paper 4–5",
    )
    lessons.solution(
        r"""**6.** Attention *is* these composed: **dot product** (Math X1) scores similarity → **softmax** (X5) turns scores into weights → a **weighted sum** of values; training it uses **cross-entropy** (M2 / X5) on the next token. Nothing new — just the pieces you already have, wired together.""",
        label="Bridge 6",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
