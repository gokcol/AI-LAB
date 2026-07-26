import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import ml_lessons2

LESSON = ml_lessons2.UNSUPERVISED


def _kmeans_once(X, k, rng, iters=100):
    """One k-means++ initialization followed by Lloyd iterations."""
    chosen = [int(rng.integers(len(X)))]
    while len(chosen) < k:
        d2 = ((X[:, None, :] - X[chosen][None, :, :]) ** 2).sum(2).min(1)
        if d2.sum() <= 0:
            remaining = np.setdiff1d(np.arange(len(X)), chosen)
            chosen.append(int(rng.choice(remaining)))
        else:
            chosen.append(int(rng.choice(len(X), p=d2 / d2.sum())))
    centroids = X[chosen].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
        assign = d.argmin(1)
        new = np.array([X[assign == j].mean(0) if np.any(assign == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new
    inertia = float(((X - centroids[assign]) ** 2).sum())
    return assign, centroids, inertia


def _kmeans(X, k, seed, iters=100, n_init=20):
    """Best of several k-means++ starts; avoids presenting one poor local run as the optimum."""
    rng = np.random.default_rng(seed)
    runs = [_kmeans_once(X, k, rng, iters) for _ in range(n_init)]
    return min(runs, key=lambda result: result[2])


st.title("M5 · Unsupervised — playground & lesson")
st.caption("No labels — find structure. Watch k-means group points and place centroids; "
           "try the wrong k and see the inertia change.")

lessons.predict(
    'As you raise **k** in k-means, does the inertia go up or down? So can you just pick k by minimizing inertia?',
    'The **globally optimal** inertia cannot rise as k increases (more centroids can fit at least as tightly), so minimizing it alone picks k = n points. This demo uses 20 k-means++ restarts to get close to that optimum; use the **elbow** or **silhouette** to choose k.',
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        pts = st.slider("points per true cluster", 20, 150, 50, 10, key="u_pts")
        true_k = st.slider("true clusters in the data", 2, 5, 3, key="u_truek")
        spread = st.slider("spread (overlap)", 0.3, 2.0, 0.8, 0.1, key="u_spread")
        k = st.slider("k (clusters to find)", 1, 6, 3, key="u_k")
        seed = st.number_input("seed", 0, 9999, 0, key="u_seed")

    rng = np.random.default_rng(int(seed))
    centers = rng.uniform(-4, 4, (int(true_k), 2))
    X = np.vstack([rng.normal(c, spread, (int(pts), 2)) for c in centers])

    assign, centroids, inertia = _kmeans(X, int(k), int(seed))

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        ax.scatter(X[:, 0], X[:, 1], c=assign, cmap="tab10", s=18, alpha=0.8,
                   vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=180,
                   edgecolors="white", linewidths=1.2, zorder=5, label="centroids")
        ax.set_title(f"k-means: found {k} clusters")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_aspect("equal", "datalim")
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("k (chosen)", int(k))
    c2.metric("true clusters", int(true_k))
    c3.metric("inertia", f"{inertia:.0f}")
    st.info("The **optimal** inertia never rises with k. A finite solver can miss that optimum, "
            "so this demo uses the best of 20 k-means++ starts rather than one fragile random "
            "initialization. Inertia still favors ever-larger k; use the **elbow** or "
            "**silhouette** to choose k.", icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlunsup")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** With 3 true blobs, **k = 2** merges two of them (higher inertia than needed); **k = 4** splits one blob (inertia keeps dropping but the extra cluster is spurious). k = 3 looks right — the elbow.

**2.** As overlap grows, k-means starts mis-assigning boundary points (it only knows distance to centroids, not the true blob shapes).

**3.** Different seeds can land in different local optima — that's k-means' **initialization sensitivity** (why k-means++ / multiple restarts exist).""",
        label="Playground 1–3",
    )
    lessons.solution(
        r"""**4.** One step: **assign** each point to its nearest centroid, then **move** each centroid to the mean of its assigned points; repeat until assignments stop changing.

**5.** Adding a centroid can only keep or shrink each point's distance to its nearest centroid, so **inertia never increases** with k (it hits 0 at k = n). Minimizing it alone just picks k = n — useless; use the elbow / silhouette.

**6.** Silhouette $=(b-a)/\max(a,b)=(3-1)/3=\mathbf{0.67}$ — comfortably inside its own cluster.""",
        label="Pencil & paper 4–6",
    )
    lessons.solution(
        r"""**7–9.** A NumPy assign/update loop + the **elbow** curve; PCA via `np.linalg.svd` with explained-variance; `sklearn`'s KMeans / DBSCAN / PCA compare on the same data.

**10.** PCA's components are the **eigenvectors of the covariance matrix** (equivalently the right singular vectors from SVD) — exactly the vectors / projection / SVD material in Math **X1**.""",
        label="Code & bridge 7–10",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
