import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

st.title("🧮 Machine Learning")
st.caption("Classical ML — same approach as the ANN module: theory + interactive playground + "
           "self-check + tasks + references, one topic at a time.")

st.subheader("📚 ML roadmap — foundations → doing it for real")
st.markdown(
    "Five levels, **basics to practice** — work top to bottom; **click any step to open it**. "
    "This is the **ML track**; the **ANN** track (single neuron → GPT) and the **Math (X1–X6)** "
    "foundations are in the sidebar **Module** selector. Every module below has an interactive "
    "playground (M7 is reference)."
)

ROADMAP = [
    ("Level 1 · Foundations",
     "What machine learning *is*, and the ideas every model shares: supervised vs. unsupervised, "
     "the **train / validation / test** split, **bias–variance**, and **over/underfitting**. "
     "Start here — it frames everything else.",
     [("views/ml_foundations.py", "M0 · Foundations", ":material/school:")],
     "**X3** probability & statistics (bias–variance, noise)."),

    ("Level 2 · Supervised learning — learn from labelled examples",
     "Predict an output from inputs given labelled data: **regression** (numbers), "
     "**classification** (categories), then two families that go non-linear — **trees & "
     "ensembles** and **SVMs & kernels**. The bread and butter of applied ML.",
     [("views/ml_regression.py", "M1 · Regression", ":material/timeline:"),
      ("views/ml_classification.py", "M2 · Classification", ":material/scatter_plot:"),
      ("views/ml_trees.py", "M3 · Trees & ensembles", ":material/account_tree:"),
      ("views/ml_svm.py", "M4 · SVM & kernels", ":material/linear_scale:")],
     "**X1** vectors / dot product · **X2** gradient descent · **X5** cross-entropy."),

    ("Level 3 · Unsupervised learning — structure without labels",
     "No labels — just find the structure: **clustering** (k-means, hierarchical, DBSCAN) and "
     "**dimensionality reduction** (PCA, t-SNE). How you discover groups and compress features.",
     [("views/ml_unsupervised.py", "M5 · Unsupervised", ":material/bubble_chart:")],
     "**X1** vectors, eigenvectors & SVD (PCA)."),

    ("Level 4 · Evaluating & choosing models",
     "How to pick a model *honestly*: **cross-validation**, **hyperparameter search**, "
     "**regularization**, **pipelines**, and the cardinal sin of **data leakage**. This is what "
     "separates a real result from a fooled-yourself one.",
     [("views/ml_model_selection.py", "M6 · Model selection", ":material/checklist:")],
     "**X3** sampling & variance of estimates."),

    ("Level 5 · Practical ML & doing it in Python",
     "The unglamorous craft that decides whether a project works: **feature engineering**, "
     "**scaling**, **missing data**, **imbalanced classes**, leakage — then assembling it all "
     "**end-to-end in scikit-learn** with live, runnable examples.",
     [("views/ml_practical.py", "M7 · Practical ML", ":material/build:"),
      ("views/ml_python.py", "M8 · ML in Python", ":material/code:")],
     "**X6** numerics · **X1** scaling & norms."),
]

for title, intro, items, math in ROADMAP:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(intro)
        for i in range(0, len(items), 3):
            row = items[i:i + 3]
            cols = st.columns(3)
            for col, (path, label, icon) in zip(cols, row):
                col.page_link(path, label=label, icon=icon)
        st.caption("➕ **Math you'll lean on:** " + math + "  *(Math module in the sidebar)*")

st.subheader("🏋 Training journey — the same loop, four models")
st.markdown("Start with a line, then a classifier. Continue in the ANN track to inspect "
            "backprop in an XOR network, then finish with next-token training. Every stop uses "
            "the same loop: **forward → loss → gradients → update**.")


def _open_ann_training(path: str):
    st.session_state["module"] = "ANN"
    st.session_state["_goto_page"] = path
    st.rerun()


journey = st.columns(4)
journey[0].page_link("views/ml_regression.py", label="1 · Train a line", icon=":material/timeline:")
journey[1].page_link("views/ml_classification.py", label="2 · Train a classifier", icon=":material/scatter_plot:")
if journey[2].button("3 · Train a network", key="ml_journey_network", icon=":material/network_node:"):
    _open_ann_training("views/mlp.py")
if journey[3].button("4 · Train a text model", key="ml_journey_text", icon=":material/smart_toy:"):
    _open_ann_training("views/transformer.py")

st.divider()

st.subheader("Where ML meets ANN")
st.markdown(r"""
The two tracks are not separate worlds — they meet at the **single neuron**:

| single neuron | + loss | = classical model |
|---|---|---|
| linear activation | MSE | **linear regression** (M1) |
| sigmoid activation | cross-entropy | **logistic regression** (M2) |

So everything you learn about weights, bias, and activations in the **ANN** module pays off
directly here — and vice-versa. Switch tracks anytime with the **Module** selector at the top
of the sidebar.
""")
