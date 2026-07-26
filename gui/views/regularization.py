"""Regularization (ANN module).

The cure for the overfitting you can trigger on the Deep-nets page. Train a high-capacity
MLP on noisy data and dial **L2 regularization (weight decay)** up and down: watch the
decision boundary go from wiggly (memorizing noise) to smooth, and use a validation set to
choose the sweet spot before inspecting the test result — the bias–variance tradeoff
(M0/M6), live.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lessons

ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _data(seed):
    X, y = make_moons(n_samples=240, noise=0.30, random_state=seed)
    Xdev, Xte, ydev, yte = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y
    )
    Xtr, Xval, ytr, yval = train_test_split(
        Xdev, ydev, test_size=0.25, random_state=1, stratify=ydev
    )
    return Xtr, Xval, Xte, ytr, yval, yte


def _model(alpha, seed):
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(40, 40), activation="relu",
                                       alpha=alpha, max_iter=2500, random_state=seed))


@st.cache_data(show_spinner=False)
def fit_alpha(alpha, seed):
    Xtr, Xval, Xte, ytr, yval, yte = _data(seed)
    clf = _model(alpha, seed).fit(Xtr, ytr)
    allX = np.vstack([Xtr, Xval, Xte]); pad = 0.6
    gx = np.linspace(allX[:, 0].min() - pad, allX[:, 0].max() + pad, 70)
    gy = np.linspace(allX[:, 1].min() - pad, allX[:, 1].max() + pad, 70)
    xx, yy = np.meshgrid(gx, gy)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    return {"Xtr": Xtr.tolist(), "ytr": ytr.tolist(),
            "Xval": Xval.tolist(), "yval": yval.tolist(),
            "Xte": Xte.tolist(), "yte": yte.tolist(),
            "gx": gx.tolist(), "gy": gy.tolist(), "Z": Z.tolist(),
            "train": clf.score(Xtr, ytr), "validation": clf.score(Xval, yval),
            "test": clf.score(Xte, yte)}


@st.cache_data(show_spinner=False)
def sweep(seed):
    Xtr, Xval, _Xte, ytr, yval, _yte = _data(seed)
    tr, val = [], []
    for a in ALPHAS:
        clf = _model(a, seed).fit(Xtr, ytr)
        tr.append(clf.score(Xtr, ytr)); val.append(clf.score(Xval, yval))
    return tr, val


_THEORY = r"""
## 1. The disease — overfitting

A high-capacity model has more than enough flexibility to **memorize** its training set —
including the random **noise**. It then scores great on training data but **fails on new
data**: training error falls to ~0 while **test** error stays high. That gap *is*
overfitting. In bias–variance terms (M0) it's **low bias, high variance** — the fit swings
wildly with whatever noise it happened to see.

## 2. Why it happens

Overfitting shows up when **capacity outruns the signal in the data**: too many parameters,
too little or too noisy data, or training too long. With 4 clean XOR points there's no noise
to fit, so capacity only helps; with **noisy** moons, a `40-40` net has ample room to wrap
around individual stray points. Note the cure isn't always "smaller model" — often it's the
**same** model, **constrained**.

## 3. The cure — penalize complexity

**Regularization** adds a penalty for complexity, so the optimizer only "spends" capacity
that genuinely lowers the loss. The workhorse is **L2** (a.k.a. ridge / **weight decay**):
add the squared size of the weights to the loss,
$$ \text{minimize}\quad L(\mathbf w) + \lambda\,\lVert\mathbf w\rVert_2^2. $$
$\lambda$ (scikit-learn's `alpha`) is the **strength dial**: $0$ = no penalty, large = weights
pushed hard toward $0$.

## 4. Why small weights = a smoother function

Big weights let a network make **steep, sharp** swings in its output — exactly the wiggles
needed to dart around noisy points. Penalizing $\lVert\mathbf w\rVert^2$ makes big weights
expensive, so the network settles for **smaller weights → gentler slopes → a smoother
boundary** that follows the trend instead of the noise. (Same mechanism as ridge regression
in M1.)

## 5. "Weight decay" — what it does to each step

Why the nickname? The penalty $\lambda\lVert\mathbf w\rVert^2$ contributes $2\lambda\mathbf w$
to the gradient, so a gradient step becomes
$$ \mathbf w \leftarrow \mathbf w - \eta\nabla L - 2\eta\lambda\mathbf w = (1-2\eta\lambda)\,\mathbf w \;-\; \eta\nabla L. $$
Every step first **shrinks** the weights by a small factor, *then* applies the usual update —
they "decay" toward zero unless the data keeps pushing back. (Modern nets use **AdamW**,
which decouples this decay cleanly — Optimizers page.)

## 6. The sweet spot (bias–variance, again)

- **Too little** $\lambda$ → wiggly boundary, big train–test gap (**overfit / high variance**).
- **Too much** $\lambda$ → weights crushed, boundary too smooth to capture the real shape
  (**underfit / high bias**).
- The best $\lambda$ sits **in between**, where *validation* accuracy peaks — chosen by a
  **validation set / cross-validation** (M6), never the test set. The **Tune** tab sweeps
  $\lambda$ and marks that peak.

## 7. L1 vs. L2

- **L2** ($\lVert\mathbf w\rVert_2^2$) shrinks *all* weights smoothly toward 0 — the default for nets.
- **L1** ($\lVert\mathbf w\rVert_1$) drives many weights to **exactly 0** → a **sparse** model
  / automatic feature selection (lasso, M1/M6); its diamond-shaped penalty makes zero-corners
  the likely solution.
- **Elastic net** blends the two.

## 8. Other regularizers (especially for nets)

- **Dropout** — randomly **zero a fraction of units each training step**, so the network
  can't lean on any single unit; it behaves like training a huge **ensemble** of sub-networks
  that share weights (averaged at test time).
- **Early stopping** — watch validation loss and **stop** when it starts rising; cheap and
  very effective.
- **Data augmentation / more data** — the most reliable cure of all: more (or cleverly
  varied) examples leave less room to memorize noise.
- **Batch norm**, smaller models, and label smoothing also regularize as a side effect.

## 9. The takeaway

Every regularizer trades a little **training fit** for a lot more **generalization** —
accepting slightly higher bias to kill variance. That trade *is* the central skill of
practical ML (M0 / M6), and **weight decay** is its everyday face in deep learning. *(Lab
tie-ins: Deep-nets page, M0 bias–variance, M6 model selection; `core/optim.py`.)*
"""

_QUIZ = [
    lessons.Question(
        "Overfitting shows up as…",
        ["high train error and high test error", "low train error but high test error (a big gap)",
         "low test error only", "identical train and test error"], 1,
        "The model memorizes training noise: train error ↓ while test error stays high — high variance."),
    lessons.Question(
        "L2 regularization (weight decay) works by…",
        ["adding more layers", "penalizing large weights, so the model prefers a smoother function",
         "increasing the learning rate", "removing the test set"], 1,
        "Adding λ‖w‖² makes big weights costly → smaller weights → smoother fit that ignores noise."),
    lessons.Question(
        "As the regularization strength λ increases from tiny to huge, validation accuracy often…",
        ["always increases", "rises to a peak (the sweet spot) then falls as the model underfits",
         "always decreases", "stays flat"], 1,
        "Too little → overfit, too much → underfit; the best λ is in between (validation picks it)."),
]

_TASKS = r"""
### In the Tune tab
1. Slide $\lambda$ to its **smallest** value — describe the boundary and the train–test
   gap. Now slide to the **largest** — what happened to the boundary, and to *both*
   train and validation accuracies?
2. Find the $\lambda$ that maximizes **validation** accuracy. Does it match the peak in the
   sweep curve?
3. Re-roll the **seed** — does the sweet-spot $\lambda$ move a lot? (That's why you tune it
   on validation, not guess.)

### Bridge
4. Connect L2 here to **ridge** (ML M1/M6), to **weight decay** in the ANN module, and to
   the bias–variance picture in **M0**.
5. Name two other regularizers you'd reach for in a deep net and when.
"""

_REFS = r"""
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 7 (regularization for deep learning).
- Hastie, Tibshirani & Friedman — *ESL*, §3.4 (ridge/lasso) & ch. 7 (model assessment).
- Srivastava et al. (2014) — *Dropout*.
- In this lab: ML **M0** (over/underfitting), **M6** (validation & model selection),
  the **Deep-nets** page (trigger the overfitting this page cures).
"""


st.title("Regularization — taming overfitting")
st.caption("Dial L2 weight decay up and down on noisy data: tune it on validation, then inspect "
           "the untouched test result.")

lessons.predict(
    "As you raise the L2 strength **λ** from 0, what happens to (a) the boundary's wiggliness and (b) the **train–validation gap**?",
    'Both often **shrink** at first: larger λ penalizes big weights → a smoother boundary → a smaller train–validation gap. Past the sweet spot it underfits. Select λ at the validation peak (green line); the test score stays out of that decision.',
)

tab_tune, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎚 Tune it", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_tune:
    st.markdown("A high-capacity MLP `(40, 40)` on **noisy two-moons**. Move the "
                "regularization strength **λ** and watch the boundary and train/validation gap:")
    cc = st.columns(2)
    alpha = cc[0].select_slider("λ  (L2 strength, sklearn alpha)", ALPHAS, value=1e-4,
                                format_func=lambda a: f"{a:g}", key="reg_a")
    seed = cc[1].slider("seed", 0, 30, 0, key="reg_seed")

    tr, val = sweep(int(seed))
    best = int(np.argmax(val))
    best_alpha = ALPHAS[best]
    r = fit_alpha(alpha, int(seed))
    m = st.columns(4)
    m[0].metric("train accuracy", f"{r['train']:.0%}")
    m[1].metric("validation accuracy", f"{r['validation']:.0%}")
    m[2].metric("test accuracy",
                f"{r['test']:.0%}" if alpha == best_alpha else "hidden",
                help="revealed only at the validation-selected λ")
    m[3].metric("train − validation gap", f"{(r['train']-r['validation']):.0%}",
                help="big gap = overfitting")

    left, right = st.columns(2)
    with left:
        st.caption("Decision boundary at this λ")
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        xx, yy = np.meshgrid(np.array(r["gx"]), np.array(r["gy"]))
        ax.contourf(xx, yy, np.array(r["Z"]), levels=20, cmap="RdBu", alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, np.array(r["Z"]), [0.5], colors="k", linewidths=1.3)
        cmap = ListedColormap(["#A32D2D", "#185FA5"])
        Xtr, ytr = np.array(r["Xtr"]), np.array(r["ytr"])
        Xval, yval = np.array(r["Xval"]), np.array(r["yval"])
        Xte, yte = np.array(r["Xte"]), np.array(r["yte"])
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cmap, s=16, edgecolors="k", linewidths=0.3)
        ax.scatter(Xval[:, 0], Xval[:, 1], c=yval, cmap=cmap, s=35, marker="s",
                   edgecolors="k", linewidths=0.5)
        if alpha == best_alpha:
            ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cmap, s=42, marker="^",
                       edgecolors="k", linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("● train  ■ validation"
                     + ("  ▲ final test" if alpha == best_alpha else "  · test hidden"),
                     fontsize=10)
        st.pyplot(fig, width="stretch")
    with right:
        st.caption("Accuracy vs λ — the sweet spot")
        fig2, ax2 = plt.subplots(figsize=(4.4, 3.9))
        ax2.semilogx(ALPHAS, tr, "-o", color="#185FA5", label="train")
        ax2.semilogx(ALPHAS, val, "-o", color="#A32D2D", label="validation")
        ax2.axvline(ALPHAS[best], color="#1D9E75", ls="--", lw=1.3)
        ax2.scatter([ALPHAS[best]], [val[best]], color="#1D9E75", zorder=5, s=60)
        ax2.axvline(alpha, color="#9C9B95", ls=":", lw=1.2)
        ax2.set_xlabel("λ (log)"); ax2.set_ylabel("accuracy")
        ax2.legend(loc="lower left", fontsize=8)
        ax2.set_title(f"best validation λ ≈ {ALPHAS[best]:g}", fontsize=10)
        st.pyplot(fig2, width="stretch")

    st.info("L2 makes large weights costly → smaller weights → a smoother boundary. Too "
            "little λ tends to overfit; too much tends to underfit, although finite validation "
            "scores can wobble. The green dashed line is the validation-selected λ; grey dotted "
            "is your current pick.",
            icon=":material/tune:")

with tab_theory:
    st.markdown(_THEORY, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="reg")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Smallest $\lambda$: a **wiggly** boundary and often a larger train–validation gap (overfit). Largest $\lambda$: the boundary goes nearly straight (over-smoothed) and train and validation accuracy usually drop (underfit).

**2.** The $\lambda$ that maximizes **validation** accuracy is the marked peak of the sweep. The test score is displayed separately but never participates in choosing $\lambda$.

**3.** Re-rolling the seed moves the sweet-spot $\lambda$ a bit — which is exactly why you **tune it on a validation set** each time rather than hard-coding a guess.""",
        label="Tune tab 1–3",
    )
    lessons.solution(
        r"""**4.** L2 here is literally **ridge** regression's penalty (ML M1/M6) and the same thing as **weight decay** in the ANN module; all three shrink weights to trade a little bias for a lot less variance — the right-hand side of the **M0** bias–variance picture.

**5.** Two more: **dropout** (randomly zero activations during training so no unit is relied on) and **early stopping** (halt when validation loss starts rising). Also useful: data augmentation, and L1 when you want *sparsity*.""",
        label="Bridge 4–5",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
