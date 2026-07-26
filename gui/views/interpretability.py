"""Interpretability — probing, ablation, and superposition (ANN module).

The lab can build a network and never ask what a particular trained one is actually doing.
This page adds the two techniques that answer that with arithmetic rather than opinion —
read a feature out of the activations (probing), and delete part of the network to see what
breaks (ablation) — and then the theory that explains why single neurons so often refuse to
mean one thing.

Deliberately scoped to what runs here. Attention-head maps and the logit lens need a saved
transformer checkpoint, and e21 saves none (and torch is kept off the server), so they are
described rather than faked. Everything on the Live tab is fitted at render time.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import lessons

st.title("Interpretability — what is this network actually doing?")
st.caption("Probe it, break it, and understand why one neuron rarely means one thing. "
           "Every number here is measured on a model trained in your browser session.")

N, K, DIM = 3000, 6, 12


@st.cache_data(show_spinner=False)
def _world(seed: int = 0):
    """Data with KNOWN generative factors, so 'did the model learn feature j' has a
    ground truth. Six independent bits drive twelve noisy observed inputs; the label is
    XOR of the first two, so bits 2-5 are real, present, and irrelevant to the task."""
    rng = np.random.default_rng(seed)
    Z = (rng.random((N, K)) < 0.5).astype(float)
    X = Z @ rng.normal(0, 1, (K, DIM)) + rng.normal(0, 0.3, (N, DIM))
    y = (Z[:, 0].astype(int) ^ Z[:, 1].astype(int))
    return X, Z, y


@st.cache_resource(show_spinner=False)
def _fit(width: int, seed: int = 0):
    X, Z, y = _world(seed)
    net = MLPClassifier((width,), max_iter=1200, random_state=0).fit(X[:2000], y[:2000])
    return net


def _hidden(net, X):
    return np.maximum(X @ net.coefs_[0] + net.intercepts_[0], 0)


def _accuracy(net, H, y):
    logits = H @ net.coefs_[1] + net.intercepts_[1]
    pred = (logits.ravel() > 0).astype(int) if logits.shape[1] == 1 else logits.argmax(1)
    return float((pred == y).mean())


tab_live, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🔬 Probe & ablate", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

# --------------------------------------------------------------------------- #
with tab_live:
    st.markdown(
        "**The setup.** Six independent hidden bits generate twelve noisy inputs. The label "
        "is `bit0 XOR bit1`, so **bits 2–5 are genuinely present in the input and completely "
        "irrelevant to the task**. Because we generated the world, we know the ground truth "
        "for every feature — which is what makes it possible to check whether an "
        "interpretability method is telling the truth."
    )
    width = st.select_slider("hidden layer width", [2, 3, 4, 6, 8, 12, 16, 24], value=16,
                             key="in_width",
                             help="The one knob that matters here. A wide layer can afford "
                                  "to carry everything; a narrow one must choose.")
    X, Z, y = _world()
    net = _fit(int(width))
    Htr, Hte = _hidden(net, X[:2000]), _hidden(net, X[2000:])
    base = _accuracy(net, Hte, y[2000:])

    st.divider()
    st.markdown("#### 1 · Linear probing — what can be *read out* of the hidden layer?")
    st.markdown("Fit a logistic regression from the hidden activations to each known bit. "
                "If it succeeds, that bit is linearly present in the representation.")

    probes = [LogisticRegression(max_iter=1000).fit(Htr, Z[:2000, j]).score(Hte, Z[2000:, j])
              for j in range(K)]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    cols = ["#2F7BEA" if j < 2 else "#C9CCD1" for j in range(K)]
    ax.bar(range(K), probes, color=cols)
    ax.axhline(0.5, color="#C0507A", ls="--", lw=1.2, label="chance")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"bit {j}" + ("\n(used)" if j < 2 else "\n(irrelevant)") for j in range(K)],
                       fontsize=8)
    ax.set_ylabel("probe accuracy"); ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)

    used = float(np.mean(probes[:2])); unused = float(np.mean(probes[2:]))
    m = st.columns(3)
    m[0].metric("task accuracy", f"{base:.3f}")
    m[1].metric("probe: bits the task uses", f"{used:.2f}")
    m[2].metric("probe: irrelevant bits", f"{unused:.2f}",
                delta=f"{unused - 0.5:+.2f} vs chance")

    if unused > 0.8:
        st.warning(
            f"**A probe succeeding does not mean the model is using that feature.** At width "
            f"{width} the irrelevant bits read out at **{unused:.2f}** — nearly perfectly — "
            f"even though the label does not depend on them at all. A wide layer has room to "
            f"carry its input along unchanged, and your probe is then reading the *input*, "
            f"not the model's reasoning. This is the standard critique of probing, and it is "
            f"why a probe result alone is evidence of **availability**, not of use. "
            f"Narrow the layer and watch the irrelevant bits fall to chance.",
            icon=":material/warning:")
    else:
        st.success(
            f"**Now the representation is selective.** At width {width} the layer cannot "
            f"afford to keep everything: the two bits the task needs still read out at "
            f"**{used:.2f}**, while the four irrelevant ones have collapsed to **{unused:.2f}**, "
            f"essentially chance. The bottleneck forced the network to *choose*, and probing "
            f"can now see what it chose.", icon=":material/check_circle:")

    st.divider()
    st.markdown("#### 2 · Ablation — what happens if we delete a unit?")
    st.markdown("Zero one hidden unit and re-measure accuracy. Probing asks what is "
                "*available*; ablation asks what is **load-bearing** — a much stronger claim.")

    drops = np.array([base - _accuracy(net, np.where(np.arange(Hte.shape[1]) == i, 0, 1) * Hte,
                                       y[2000:]) for i in range(Hte.shape[1])])
    order = np.argsort(drops)[::-1]
    fig2, ax2 = plt.subplots(figsize=(7.2, 2.7))
    ax2.bar(range(len(drops)), drops[order], color="#1D9E75")
    ax2.axhline(0, color="#33312E", lw=1)
    ax2.set_xticks(range(len(drops)))
    ax2.set_xticklabels([str(i) for i in order], fontsize=7)
    ax2.set_xlabel("hidden unit (sorted by importance)")
    ax2.set_ylabel("accuracy lost when ablated")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout(); st.pyplot(fig2)

    m2 = st.columns(3)
    m2[0].metric("most important unit", f"#{order[0]}", delta=f"-{drops[order[0]]:.3f}",
                 delta_color="inverse")
    m2[1].metric("units costing < 0.5 %", f"{int((drops < 0.005).sum())} of {len(drops)}")
    m2[2].metric("total if summed", f"{drops.sum():.3f}",
                 help="Compare with the accuracy actually available to lose. Sums far above "
                      "it mean the units are redundant; far below means no single unit "
                      "matters, which is what distribution looks like.")
    st.info(
        f"**Read the two charts against each other.** Probing said {int(sum(p > 0.8 for p in probes))} "
        f"of {K} features are readable; ablation says {int((drops > 0.01).sum())} of "
        f"{len(drops)} units are load-bearing at the 1 % level. Those are different questions "
        f"and they give different answers. The individual ablation drops also do not sum to "
        f"the total accuracy — the units are not independent contributors, which is the "
        f"first symptom of the distributed representation the Theory tab is about.",
        icon=":material/compare_arrows:")

# --------------------------------------------------------------------------- #
with tab_theory:
    st.markdown(r"""
## 1. The question this page exists for

You can now build a network, train it, and measure that it works. "What is *this* trained
network doing?" is a different question, and nothing so far in the lab answers it. That gap
matters more than it used to: the same object is now asked to make decisions people care
about, and "it scored well on the test set" is not an account of anything.

There are two honest tools available without new infrastructure, and one important idea
that explains why the job is hard.

## 2. Probing — is the feature *there*?

Take the activations of a hidden layer and fit a simple classifier — usually logistic
regression — from those activations to some property you care about. If a **linear** probe
succeeds, the property is linearly available in the representation.

The discipline is in the caveats, and the Live tab demonstrates all of them:

- **Availability is not use.** A wide layer can pass its input through almost unchanged, so
  a probe may be reading the *input*, not the model's computation. On the Live tab, at width
  16 every irrelevant bit reads out at 0.95+; narrow the layer to 4 and those same bits fall
  to chance while the task-relevant ones stay at 0.99. Same probe, same data, opposite
  conclusion — the difference is entirely whether the model had room to be lazy.
- **A strong enough probe finds anything.** With a nonlinear probe of sufficient capacity you
  are measuring your probe, not the model. This is why linear probes are the convention.
- **You need a control.** Probe accuracy is meaningless without comparing against the same
  probe on the raw input, or on a randomly-initialised network of the same shape.

## 3. Ablation — is the feature *load-bearing*?

Set a unit (or head, or layer) to zero and measure what the model loses. This asks a
causal question rather than a correlational one, and it is correspondingly stronger
evidence. The usual refinement is to replace the activation with its **mean** over the
dataset rather than with zero, so you remove the *information* it carries without also
knocking the layer off its normal operating range.

Two things you will see immediately on the Live tab:

- Most single units cost almost nothing. Deleting them individually is nearly free.
- The individual costs do not add up to the whole. If every unit is worth ~1 % but the model
  is worth 100 %, the accounting is wrong somewhere — and it is wrong because the units are
  not independent contributors.

That second observation is the door into the real subject.

## 4. Distributed representation — the 1986 half of the story

The lab already teaches this in **Embeddings**: Hinton's family-tree network learned
concepts like *nationality* and *generation* across a six-unit bottleneck, with no single
unit standing for any one of them. A concept is a **direction in activation space**, not a
neuron. That is why ablating one unit degrades everything slightly instead of removing one
capability cleanly, and it has been understood since 1986.

Here is what was *not* obvious in 1986, and is the modern half.

## 5. Superposition — why a neuron means several things

Real networks appear to represent **more features than they have neurons**. The neurons are
**polysemantic**: one unit fires for several unrelated things.

The mechanism is a fact you have already measured, in Math **X1 §17**. In $d$ dimensions
there are only $d$ mutually orthogonal directions — but there are *exponentially many*
**nearly** orthogonal ones. Two random unit vectors in 1000-D have $\cos\theta$ of
$0.00\pm0.032$. So a network can assign thousands of features their own direction in a
1000-dimensional space, accepting a small interference term between each pair, and come out
ahead.

The trade is only worth taking under **sparsity**: if features are rarely active at the
same time, the interference rarely materialises. Natural data is extremely sparse in this
sense — of everything a language model could be representing, almost none of it is relevant
to any one token. So the model packs, and the cost is that no neuron has a clean meaning.

This reframes the failure of naive interpretability. "Neuron explanations look inconsistent"
is not a measurement problem to be solved with more careful staring. It is the predicted
consequence of a strategy the network is using deliberately, because it is efficient.

## 6. Which is why sparse autoencoders exist

If features live in directions rather than neurons, and are packed in superposition, then
the task is to **recover the directions**. Train a wide autoencoder on the activations —
much wider than the layer itself — with a sparsity penalty forcing only a few of its units
to fire at once:

$$ \min \; \lVert a - \hat a\rVert^2 \;+\; \lambda \lVert h \rVert_1, \qquad \hat a = W_{\text{dec}}h,\; h = \mathrm{relu}(W_{\text{enc}}a + b) $$

The reconstruction term keeps the information; the $\ell_1$ term (Math **X4**, and the same
sparsity-inducing penalty as Lasso in **M4 · Regularization**) forces it into few active
directions. What comes out are dictionary elements that are far more monosemantic than the
original neurons — one direction, one recognisable concept.

That is the current mainstream approach, and it is worth being honest about its state: the
dictionaries are large, incomplete, sensitive to hyperparameters, and there is active
disagreement about how much of the model they truly explain. It is a live research
programme, not a solved tool.

## 7. What this page cannot show you, and why

Attention-head visualisation and the logit lens are the other two standard techniques, and
they are described here rather than demonstrated. The reason is concrete: `e21_nanogpt`
never saves a checkpoint, and PyTorch is deliberately kept out of `requirements.txt` so the
public server can run without it. The GUI's own "Tiny GPT" page trains an sklearn
`MLPClassifier` over one-hot character windows — it has no attention heads to plot. Faking
either with pretty pictures of a model that is not there would be worse than the omission.

*(Related: **Embeddings** §9 for distributed representations, Math **X1 §17** for the
near-orthogonality that makes superposition possible, **M4** for the $\ell_1$ penalty.)*
""")

# --------------------------------------------------------------------------- #
with tab_quiz:
    lessons.render_quiz([
        lessons.Question(
            "A linear probe reads a feature out of a hidden layer at 97 % accuracy. What "
            "have you established?",
            ["The model uses that feature to make its decision",
             "The feature is linearly available in that layer — which is not the same as the "
             "model using it",
             "The feature is stored in a single neuron",
             "Nothing at all"], 1,
            "Availability, not use. A wide layer can pass its input through nearly unchanged, "
            "in which case the probe is reading the input. Narrow the layer on the Live tab "
            "and irrelevant features drop to chance."),
        lessons.Question(
            "Why is ablation stronger evidence than probing?",
            ["It is faster to compute",
             "It asks a causal question — remove this and see what breaks — rather than a "
             "correlational one",
             "It works on larger models",
             "It needs no ground-truth labels"], 1,
            "Probing correlates activations with a property. Ablation intervenes and measures "
            "the consequence."),
        lessons.Question(
            "On the Live tab the individual ablation costs do not sum to the model's total "
            "accuracy. What does that indicate?",
            ["A bug in the measurement",
             "The units are not independent contributors — the representation is distributed, "
             "so capability is spread across many units",
             "The model is overfitted",
             "Ablation should use the mean instead of zero"], 1,
            "It is the first observable symptom of distributed representation, and the reason "
            "single-neuron explanations disappoint."),
        lessons.Question(
            "What makes superposition possible in the first place?",
            ["Networks have more neurons than features",
             "In d dimensions there are exponentially many NEARLY orthogonal directions even "
             "though only d exactly orthogonal ones",
             "Weights are initialised randomly",
             "ReLU discards half the signal"], 1,
            "The same geometry measured in Math X1 §17: cos θ between random unit vectors "
            "concentrates at 0 with spread 1/√d, so near-orthogonal directions are abundant."),
        lessons.Question(
            "Superposition only pays off when features are…",
            ["Numerous", "Sparse — rarely active at the same time, so the interference between "
             "their directions rarely materialises", "Binary", "Independent"], 1,
            "Sparsity is the condition. Dense features would collide constantly and the "
            "packing would cost more than it saves."),
        lessons.Question(
            "Why does the sparse-autoencoder objective need BOTH a reconstruction term and an "
            "ℓ1 penalty?",
            ["To speed up training",
             "Reconstruction keeps the information; the ℓ1 term forces it into few active "
             "directions, which is what makes them monosemantic",
             "The ℓ1 term prevents overfitting",
             "To make the autoencoder invertible"], 1,
            "Reconstruction alone gives you a rotated copy of the activations. The sparsity "
            "penalty is what turns 'some basis' into 'an interpretable one'."),
    ], prefix="interp")

# --------------------------------------------------------------------------- #
with tab_tasks:
    st.markdown("#### Tasks")
    st.markdown(
        "1. Set the width to 16 and read the probe chart. Now set it to 4. State, in one "
        "sentence, what changed about what the probe is measuring — the probe did not change.\n"
        "2. At width 2 the task-relevant bits also start to degrade. What does that tell you "
        "about the difference between *the model can solve the task* and *the features are "
        "linearly readable*?\n"
        "3. Ablate the single most important unit. Compare that drop with the sum of all the "
        "drops. Explain the discrepancy in terms of §4.\n"
        "4. Design a control that would distinguish 'the probe is reading the model's "
        "computation' from 'the probe is reading the input'. (There are at least two.)\n"
        "5. A colleague says a neuron is 'the dog neuron' because it fires on dog images. "
        "Give the two experiments you would ask for before believing it."
    )
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Nothing about the probe changed; what changed is what the *layer* contains.
At width 16 the network has enough capacity to carry its whole input forward, so the probe
reads the input through the model. At width 4 the layer is a bottleneck, the network must
discard whatever the task does not need, and the irrelevant bits fall to chance. Probe
accuracy is a property of the representation, not of the model's reasoning.

**2.** They come apart. At width 2 task accuracy stays at ~1.00 while the probes for bits 0
and 1 drop to ~0.75. The network still computes `bit0 XOR bit1` perfectly — it just no
longer stores the two bits *separately and linearly*; it stores the XOR. A linear probe
cannot read either input bit out of a representation that encodes only their parity, which
is a good reminder that a failed probe is not evidence of a missing capability.

**3.** The most important unit typically costs ~10 %, while the summed drops usually exceed
what is available to lose, and no single unit is near the total. Both facts point the same
way: capability is spread across units with heavy redundancy, so removing one leaves the
others to compensate. That is distributed representation (§4) seen from the causal side.""",
        label="Live 1–3")
    lessons.solution(
        r"""**4.** Two standard controls. (a) **Probe the raw input** with the same classifier:
if it does as well or better, the hidden layer has added nothing and you are reading the
input. (b) **Probe a randomly-initialised network of identical shape**: an untrained net
still performs a random projection, which preserves a great deal of linearly-decodable
information — anything the random net matches is not something training produced. A third,
stronger option is to ablate the direction the probe found and check that the model's
behaviour changes.

**5.** First, **ablation**: remove the unit (or better, the direction) and check that dog
recognition specifically degrades while other classes hold. Second, **a search for
polysemanticity**: sweep a large and *diverse* input set for the unit's top activations —
if it also fires hard on car grilles and certain textures, it is a polysemantic unit in
superposition (§5) and "the dog neuron" is a story told about its top-9 images. Any claim
about a single neuron needs both the causal test and the negative search.""",
        label="Concept 4–5")

# --------------------------------------------------------------------------- #
with tab_ref:
    st.subheader("Reading & references")
    st.markdown("""
- Alain & Bengio (2016) — *Understanding intermediate layers using linear classifier probes*
  (the method, and its limits).
- Hewitt & Liang (2019) — *Designing and Interpreting Probes with Control Tasks* — why a
  probe without a control measures your probe.
- Belinkov (2022) — *Probing Classifiers: Promises, Shortcomings, and Advances* — a survey
  that is unusually honest about what probing does not show.
- Elhage et al. (2022) — *Toy Models of Superposition* (Anthropic) — the paper that made
  superposition concrete and measurable.
- Olah et al. (2020) — *Zoom In: An Introduction to Circuits*.
- Bricken et al. (2023) — *Towards Monosemanticity* — sparse autoencoders on a real model.
- Templeton et al. (2024) — *Scaling Monosemanticity* — the same at production scale.
- Hinton, Rumelhart & Williams (1986) — the family-tree network; distributed representation
  as originally demonstrated (see the **Embeddings** page).

**In this lab:** near-orthogonality in high dimensions — Math **X1 §17** · distributed
representations — **Embeddings** §9 · the ℓ1 penalty — **M4 · Regularization** and Math
**X4**.

**Caveat.** Interpretability is the least settled subject in this lab. The techniques here
are real and the demonstrations are honest, but the field disagrees with itself about how
much any of them explain. Treat conclusions as provisional in a way that is not necessary
for, say, backpropagation.
""")
