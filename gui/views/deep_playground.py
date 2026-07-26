"""Deep nets on 2-D data (ANN module).

The MLP page trains the lab's own engine on XOR; here the *same idea* — a multilayer
perceptron with a nonlinearity — is scaled up (via scikit-learn's fast MLP) to real 2-D
shapes: two moons, concentric circles, a spiral. Watch the decision boundary curve to fit
as you change the architecture and activation.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lessons

ARCHS = {"1 × 8": (8,), "1 × 32": (32,), "2 × 16": (16, 16), "3 × 24": (24, 24, 24)}


def _spiral(n, noise, seed):
    rng = np.random.default_rng(seed)
    k = n // 2
    t = np.sqrt(rng.random(k)) * 2.6 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)]
    b = np.c_[-t * np.cos(t), -t * np.sin(t)]
    X = np.vstack([a, b]) + noise * 2.2 * rng.standard_normal((2 * k, 2))
    X = X / np.abs(X).max() * 3.0
    return X, np.r_[np.zeros(k), np.ones(k)].astype(int)


def _dataset(name, noise, seed):
    if name == "Two moons":
        return make_moons(n_samples=260, noise=noise, random_state=seed)
    if name == "Concentric circles":
        return make_circles(n_samples=260, noise=noise, factor=0.45, random_state=seed)
    return _spiral(260, noise, seed)


@st.cache_data(show_spinner=False)
def fit(name, arch_key, activation, noise, seed):
    X, y = _dataset(name, noise, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=ARCHS[arch_key], activation=activation,
                                      alpha=1e-4, max_iter=2500, random_state=seed))
    clf.fit(Xtr, ytr)
    pad = 0.6
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 70)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 70)
    xx, yy = np.meshgrid(gx, gy)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    return {
        "Xtr": Xtr.tolist(), "ytr": ytr.tolist(), "Xte": Xte.tolist(), "yte": yte.tolist(),
        "gx": gx.tolist(), "gy": gy.tolist(), "Z": Z.tolist(),
        "train": clf.score(Xtr, ytr), "test": clf.score(Xte, yte),
    }


_THEORY = r"""
## 1. Beyond XOR — curved boundaries

XOR (MLP page) is the smallest non-linear problem — four points, two hidden units. The
*same* machine — a hidden layer plus a nonlinearity — scales straight up to **moons, rings,
and spirals**. With enough hidden capacity an MLP can bend its boundary into essentially
**any** shape (universal approximation, MLP §5). This page swaps the lab's scalar engine for
scikit-learn's **fast, vectorized MLP**, so you can train on hundreds of points in real time
and *watch* the boundary form.

## 2. How hidden units carve up space

Picture one **ReLU** hidden unit: it's a line — "off" (0) on one side, rising linearly on
the other — i.e. a **fold** in the input space. The output neuron **adds up** many such
folds. A handful makes a polygon; dozens approximate a smooth curve; arranged right they
enclose a ring or trace a spiral. So **more units = more folds = more intricate regions** —
which is exactly what you see when you switch `1×8 → 2×16`.

## 3. Depth vs. width

- **Width** (units per layer) adds folds at one level — more pieces in the boundary.
- **Depth** (more layers) lets the network **compose** features: layer 1's folds become the
  *inputs* to layer 2, which folds *those*. So depth builds shape **hierarchically** and can
  represent some boundaries with far fewer total units than one giant wide layer. (Vision:
  pixels → edges → textures → parts → objects.)

A spiral is the good stress test here — it usually needs real width *or* a second layer.

## 4. Activation shapes the boundary

- **ReLU** — piecewise-linear, so boundaries look **polygonal** (straight segments); cheap,
  non-saturating, the deep-learning default.
- **tanh / logistic** — smooth, so boundaries look **curvy**; but they **saturate** and can
  vanish gradients in deep stacks (ANN §6).

Toggle them on the same data — the boundary's *texture* changes even when accuracy is alike.

## 5. Boundary vs. regions vs. confidence

The black line is where the model is 50/50 ($p=0.5$); the **shading** is the predicted
probability — vivid where it's confident, washed-out near the boundary and in regions with
no data. Far from the points the prediction is essentially a guess: a model only really
"knows" where it had examples.

## 6. Data, noise & overfitting

Capacity cuts both ways. Turn the **noise** up and a big network can start **tracing the
noise** — a wiggly boundary that nails the training dots but misreads new ones. That's
**overfitting**: training accuracy ≫ **test** accuracy (the held-out ▲ points, M0). The page
flags it when the gap grows; the cure is **regularization** (next page) — or more data.

## 7. Why we standardize first

The pipeline runs **`StandardScaler`** before the MLP, because neural nets train far better
when inputs are centered and comparably scaled (same reason as M7 / Math X-scaling). Without
it, a feature with a larger numeric range dominates and training is slower and less stable.

## 8. Same recipe underneath

Nothing new is happening: it's still **forward → loss → backward (backprop) → optimizer
step**, repeated, just on more data with a tuned solver. The takeaway: **depth + a
nonlinearity learns features** that make even tangled classes separable by the final layer —
the core trick that, scaled to images and text, *is* modern deep learning. *(Lab tie-ins:
MLP, Backprop, Optimizers, M0, M7.)*
## 9. Where the U-curve stops applying — double descent

The prediction at the top of this page says test accuracy **rises then falls** as you add
capacity, and on this page, with this much data, it does. Meanwhile the Tiny GPT page
teaches **scaling laws** and Chinchilla: make the model bigger and, given the tokens to
match, it gets predictably *better*. Both are in this lab, and read together they look
like a contradiction. They are not — but the boundary between them is worth knowing,
because it is where classical ML stops describing modern deep learning.

**The classical U-curve is correct in the regime it was drawn for**: the model has fewer
parameters than the data has constraints, so there is one best fit, and capacity beyond the
true complexity of the signal buys only noise-fitting. Polynomial degree, tree depth, $k$
in $k$-NN, the width slider on this page against a small noisy dataset — all genuinely
U-shaped.

**What changes is what happens when you keep going past the point of exactly fitting the
data.** Let $P$ be the number of parameters and $n$ the number of training points. At
$P = n$ there is exactly *one* function that interpolates the training set, and no freedom
left to choose a sensible one — so the fit is wildly contorted between the points, and test
error **spikes**. Push past it and there are now infinitely many interpolating solutions;
gradient descent does not pick one at random, it converges to the **minimum-norm** one,
which is the smoothest of them. More parameters means a richer set to choose the smooth
one from, and test error falls *again* — often below anything the underparameterised regime
achieved.

$$ \text{test error} \;\;\text{rises to a peak at } P \approx n,\;\text{ then descends a second time} $$

The measured shape, from the demo on this page — ridgeless regression on random ReLU
features, $n=40$ training points:

| features $P$ | 5 | 20 | 38 | **40** | 50 | 120 | 2000 |
|---|---|---|---|---|---|---|---|
| test MSE | 0.94 | 1.20 | 4.78 | **32.3** | 1.66 | 0.33 | **0.14** |

The classical story is the left half of that table and it is accurate there. But the best
underparameterised model scores 0.94, and the model with **fifty times more parameters than
data points** scores 0.14 — nearly seven times better. No amount of U-curve reasoning
predicts that.

**So which page is right?** Both, in their regime:

- This page trains a *small* net on a *small* noisy 2-D dataset, and the width slider stops
  well short of $P \gg n$. You are on the left of the peak; the U-curve is what you will see,
  and the prediction is correct.
- A frontier language model has $P \gg n$ by a wide margin and never leaves the second
  descent. That is why "make it bigger" is a strategy there and malpractice here.

**The honest caveats.** Double descent is real and reproducible, but it is not a licence to
skip regularisation: with well-tuned ridge or early stopping the peak flattens and the curve
can become monotone — the spike is partly an artifact of interpolating *exactly*. It also
depends on how capacity is counted; the peak sits at $P\approx n$, not at some absolute
size. And "more parameters is better" only holds with the data and compute to match, which
is precisely what Chinchilla quantifies.

*(Related: the classical U-curve in **M3 · Model selection**, scaling laws in **Level 5 ·
Tiny GPT** §11, and the minimum-norm idea in Math **X1**.)*
"""

_QUIZ = [
    lessons.Question(
        "Why can an MLP separate two interleaved spirals when a line cannot?",
        ["it uses more data", "hidden units + a nonlinearity bend the boundary into an arbitrary curved shape",
         "it removes outliers", "spirals are actually linear"], 1,
        "With enough hidden capacity an MLP approximates any boundary — straight lines need not apply."),
    lessons.Question(
        "Increasing hidden width/depth mainly increases…",
        ["the learning rate", "model capacity — the complexity of boundary it can represent (and the overfitting risk)",
         "the number of classes", "the input dimension"], 1,
        "More units/layers = more capacity; great until it starts fitting noise (→ regularization)."),
    lessons.Question(
        "ReLU vs tanh boundaries tend to look…",
        ["identical", "ReLU polygonal (piecewise-linear), tanh smooth/curvy",
         "ReLU circular, tanh square", "both perfectly straight"], 1,
        "ReLU is piecewise-linear so regions are polygonal; tanh is smooth so boundaries curve."),
]

_TASKS = r"""
### In the Train tab
1. On **Two moons**, compare `1 × 8` vs `2 × 16` — does the extra capacity give a cleaner
   boundary, or just more wiggle?
2. Switch to **Spiral** — what's the smallest architecture that separates it? Watch train
   vs **test** accuracy as you add capacity.
3. Toggle **ReLU ↔ tanh** on the same data — describe how the boundary's *texture* changes.
4. Crank the **noise** up — when does even a big network's test accuracy fall? (overfitting.)

### Bridge
5. Connect to **MLP** (the same net on XOR, on the lab's engine), **Optimizers** (how it's
   trained), and **M0/M7** (capacity vs. overfitting).
"""

_REFS = r"""
- **TensorFlow Playground** (playground.tensorflow.org) — the classic interactive version of this.
- scikit-learn — [neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).
- Nielsen — *Neural Networks and Deep Learning*, ch. 4 (a visual proof MLPs can fit any function).
- In this lab: **MLP**, **Backprop**, **Optimizers**, and ML **M0** (over/underfitting).
"""


st.title("Deep nets on 2-D data")
st.caption("The MLP idea scaled past XOR — watch a multilayer perceptron bend its boundary "
           "to fit moons, rings and spirals (scikit-learn's fast MLP).")

lessons.predict(
    "Crank the network's width/depth *way* up on the noisy data — does **test** accuracy keep rising, plateau, or fall? Why?",
    'It rises, then **falls**. Extra capacity first captures the true shape, then starts memorizing noise (**overfitting**): train accuracy keeps climbing while test accuracy drops. The goal is *enough* capacity, not maximum.',
)


def _double_descent():
    """Ridgeless regression on random ReLU features: the cleanest reproduction of double
    descent that runs in a second. Minimum-norm (pinv) is essential — the second descent
    is a property of WHICH interpolating solution you land on, so a regularised solve
    would flatten the very peak this is here to show."""
    st.markdown("**The U-curve, and what is on the other side of it.** Test error against "
                "model size, all the way past the point where the model can fit the "
                "training data exactly. The dashed line is $P=n$, where there is exactly "
                "one interpolating solution and no freedom to pick a good one.")
    c = st.columns(3)
    n = c[0].slider("training points $n$", 20, 120, 40, 5, key="dd_n")
    noise = c[1].slider("label noise σ", 0.0, 1.0, 0.35, 0.05, key="dd_noise")
    trials = c[2].slider("repeats (median)", 5, 40, 20, 5, key="dd_trials",
                         help="The peak is violent and heavy-tailed; the median over "
                              "repeats is what makes the shape legible.")
    d = 20
    Ps = sorted({5, 10, 20, 30, int(0.9 * n), int(0.95 * n), n, int(1.05 * n) + 1,
                 int(1.25 * n), 2 * n, 3 * n, 6 * n, 15 * n, 40 * n})
    Ps = [P for P in Ps if P >= 2]

    def run(P):
        errs = []
        for t in range(trials):
            r = np.random.default_rng(1000 + t)
            beta = r.normal(0, 1, d) / np.sqrt(d)
            Xtr, Xte = r.normal(0, 1, (n, d)), r.normal(0, 1, (400, d))
            ytr = Xtr @ beta + r.normal(0, noise, n)
            yte = Xte @ beta
            W = r.normal(0, 1, (d, P)) / np.sqrt(d)
            w = np.linalg.pinv(np.maximum(Xtr @ W, 0)) @ ytr     # minimum-norm fit
            errs.append(float(np.mean((np.maximum(Xte @ W, 0) @ w - yte) ** 2)))
        return float(np.median(errs))

    te = [run(P) for P in Ps]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.loglog(Ps, te, "o-", color="#2F7BEA", lw=2, ms=4)
    ax.axvline(n, color="#C0507A", ls="--", lw=1.4, label=f"$P=n={n}$ (interpolation)")
    ax.set_xlabel("parameters $P$  (random ReLU features)")
    ax.set_ylabel("test MSE"); ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)

    under = min(t for P, t in zip(Ps, te) if P < n)
    peak = max(te)
    over = te[-1]
    m = st.columns(3)
    m[0].metric("best classical (P < n)", f"{under:.3f}")
    m[1].metric("at the interpolation peak", f"{peak:.2f}", delta=f"×{peak/under:.0f} worse",
                delta_color="inverse")
    m[2].metric(f"heavily overparameterised (P={Ps[-1]})", f"{over:.3f}",
                delta=f"×{under/over:.1f} better", delta_color="normal")
    if over < under:
        st.success(
            f"**The second descent is real.** The best model with fewer parameters than data "
            f"scores {under:.3f}. Push to {Ps[-1]} parameters — {Ps[-1]//n}× more than there "
            f"are training points — and it scores {over:.3f}, **{under/over:.1f}× better**, "
            f"having passed through a peak {peak/under:.0f}× worse on the way. The U-curve "
            f"describes only the left of that dashed line.", icon=":material/trending_down:")
    else:
        st.info("At this noise level the peak is shallow and the second descent has not "
                "overtaken the classical regime — raise the noise, or lower n, to sharpen it.",
                icon=":material/lightbulb:")
    st.caption("Ridgeless (minimum-norm) fits on purpose: the second descent is about *which* "
               "interpolating solution gradient descent finds, so adding ridge flattens the "
               "peak — which is itself the point, and the reason this is not a licence to "
               "drop regularisation.")


tab_train, tab_dd, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🌀 Train it", "📉 Double descent", "📖 Theory", "❓ Self-check", "🛠 Tasks",
     "📚 References"]
)

with tab_train:
    c = st.columns(4)
    name = c[0].selectbox("dataset", ["Two moons", "Concentric circles", "Spiral"], key="dp_ds")
    arch_key = c[1].selectbox("hidden layers", list(ARCHS), index=2, key="dp_arch")
    activation = c[2].selectbox("activation", ["relu", "tanh", "logistic"], key="dp_act")
    noise = c[3].select_slider("noise", [0.05, 0.1, 0.2, 0.3], value=0.2, key="dp_noise")
    seed = st.slider("seed", 0, 30, 0, key="dp_seed")

    r = fit(name, arch_key, activation, noise, int(seed))

    m = st.columns(2)
    m[0].metric("train accuracy", f"{r['train']:.0%}")
    m[1].metric("test accuracy", f"{r['test']:.0%}", help="held-out 30% — the honest score")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    xx, yy = np.meshgrid(np.array(r["gx"]), np.array(r["gy"]))
    ax.contourf(xx, yy, np.array(r["Z"]), levels=20, cmap="RdBu", alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, np.array(r["Z"]), levels=[0.5], colors="k", linewidths=1.3)
    cmap = ListedColormap(["#A32D2D", "#185FA5"])
    Xtr, ytr = np.array(r["Xtr"]), np.array(r["ytr"])
    Xte, yte = np.array(r["Xte"]), np.array(r["yte"])
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cmap, s=18, edgecolors="k", linewidths=0.3)
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cmap, s=42, marker="^",
               edgecolors="k", linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("● train  ▲ test  ·  boundary = where p = 0.5", fontsize=10)
    st.pyplot(fig, width="stretch")

    gap = r["train"] - r["test"]
    if gap > 0.12:
        st.warning(f"Train beats test by {gap:.0%} — the network is **overfitting** (tracing "
                   "noise). Fewer units or **regularization** (next page) would help.",
                   icon=":material/warning:")
    else:
        st.success("Train and test agree — the boundary captures the real shape, not the noise.",
                   icon=":material/check_circle:")
    st.info("Same MLP idea as the XOR page, scaled to real 2-D data with scikit-learn's "
            "vectorized solver. Capacity (width/depth) lets it bend the boundary; too much "
            "and it overfits.", icon=":material/blur_on:")

with tab_dd:
    _double_descent()

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="deepplay")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** `2 × 16` gives a cleaner boundary on two-moons *up to a point*; past what the data needs, the extra capacity just adds wiggle (it starts tracing noise).

**2.** The spiral needs real depth/width — roughly `2 × 16` or more before it separates cleanly. As you add capacity, watch **test** accuracy plateau then dip while **train** keeps rising: that gap is overfitting.

**3.** **ReLU** gives a piecewise-linear, faceted boundary (made of straight kinks); **tanh** gives a smoother, curved one — same data, different function class.

**4.** Crank the noise up and even a big network's **test** accuracy falls: with enough capacity it memorizes the noisy points (train stays high, test drops) — the definition of overfitting.""",
        label="Train tab 1–4",
    )
    lessons.solution(
        r"""**5.** It's the same MLP as the **MLP** page (there on XOR, on the lab's own engine), trained by the ideas on the **Optimizers** page, and it *is* the bias–variance / capacity story made visual — **M0** (over/underfitting) and **M7** (practical control of it).""",
        label="Bridge 5",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
