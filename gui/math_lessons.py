"""Math module lesson content. Reuses the Lesson/Question/quiz engine from lessons.py."""

from __future__ import annotations

from lessons import Lesson, Question

_THEORY = r"""
Linear algebra is the **language of machine learning**: a data point is a vector, a layer
is a matrix multiply, attention is a product of matrices, and PCA is an eigen-decomposition.
Master this page and the rest of the lab reads like sentences instead of symbols.

## 1. What a vector is

An ordered list of numbers, $\mathbf x=(x_1,\dots,x_n)\in\mathbb R^n$. Two readings:
a **point** in $n$-dimensional space, or an **arrow** from the origin. The number of
entries $n$ is the **dimension**. In ML *everything* is a vector — a data sample (its
features), a neuron's weights, a word's embedding, an image (pixels flattened).

## 2. Vector operations & geometry

- **Addition** $(\mathbf a+\mathbf b)_i=a_i+b_i$ — arrows tip-to-tail.
- **Scalar multiply** $c\mathbf a$ — scales length ($c<0$ flips direction).
- **L2 norm (length)** $\lVert\mathbf a\rVert=\sqrt{\sum_i a_i^2}$ — the arrow's length.
- **Unit vector** $\hat{\mathbf a}=\mathbf a/\lVert\mathbf a\rVert$ — direction only.
- **Distance** between two points is $\lVert\mathbf a-\mathbf b\rVert$ — the ruler of k-NN, k-means, and RBF kernels.

## 3. The dot product — the star of the show

Two equivalent definitions, one algebraic and one geometric:

$$ \mathbf a\cdot\mathbf b=\sum_i a_i b_i=\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert\cos\theta. $$

It measures **alignment**: large positive when the vectors point the same way, $0$
when perpendicular, negative when opposed. Key properties: it is **symmetric**
($\mathbf a\cdot\mathbf b=\mathbf b\cdot\mathbf a$), **distributes** over addition, and
$\mathbf a\cdot\mathbf a=\lVert\mathbf a\rVert^2$.

- **Cosine similarity** $\cos\theta=\dfrac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$
  — similarity of *direction*, independent of length.
- **Projection** of $\mathbf b$ onto $\mathbf a$ has length $\dfrac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert}$
  — "how much of $\mathbf b$ lies along $\mathbf a$".
- **Orthogonal** $\iff \mathbf a\cdot\mathbf b=0$.

Geometrically, the projection is **b's shadow along a** — drop a perpendicular from the
tip of b onto the line of a:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 380 250" style="width:100%;max-width:380px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two vectors a and b drawn from the origin with the angle theta between them. The projection of b onto a is b's shadow along a, found by dropping a perpendicular from the tip of b to the line of a."><defs><marker id="x1aa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#185FA5"/></marker><marker id="x1bb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#C0507A"/></marker></defs><rect x="1" y="1" width="378" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="60" y1="200" x2="225" y2="145" stroke="#1D9E75" stroke-width="6" opacity="0.35"/><line x1="60" y1="200" x2="298" y2="121" stroke="#185FA5" stroke-width="2.5" marker-end="url(#x1aa)"/><line x1="60" y1="200" x2="198" y2="71" stroke="#C0507A" stroke-width="2.5" marker-end="url(#x1bb)"/><line x1="200" y1="70" x2="225" y2="145" stroke="#9C9B95" stroke-width="1.4" stroke-dasharray="4 3"/><path d="M 110 185 A 48 48 0 0 0 98 167" fill="none" stroke="#6B6A66" stroke-width="1.3"/><circle cx="60" cy="200" r="3" fill="#33312E"/><g font-family="sans-serif" font-size="14"><text x="305" y="119" fill="#185FA5">a</text><text x="188" y="62" fill="#C0507A">b</text></g><g font-family="sans-serif" font-size="11"><text x="113" y="181" fill="#6B6A66">θ</text><text x="196" y="192" fill="#0E5E45">projection of b onto a</text></g></svg></div>

## 4. Why the dot product runs ML & ANN

- A neuron's pre-activation **is** a dot product: $z=\mathbf w\cdot\mathbf x+b$. The
  neuron fires most when the input *aligns* with its weights — the "matched filter" view.
- **Embeddings, search & RAG:** words/documents become vectors; relevance = cosine
  similarity; "find similar" = largest dot products.
- **Attention** scores every token pair with a dot product (query · key).
- **Kernels** (SVM, M4) replace $\mathbf x\cdot\mathbf x'$ with $K(\mathbf x,\mathbf x')$ — a dot product in a richer space.

## 5. From vectors to matrices — a layer is a batch of dot products

A whole layer applies many dot products at once: $\mathbf z = W\mathbf x + \mathbf b$,
where each **row of $W$ is one neuron's weight vector**. Matrix multiplication is a
batch of dot products — the single most-executed operation in deep learning, and exactly
what GPUs accelerate.

## 6. Matrix multiplication, precisely

Entry $(i,j)$ of $AB$ is **row $i$ of $A$ dotted with column $j$ of $B$**:
$(AB)_{ij}=\sum_k A_{ik}B_{kj}$. The **inner dimensions must match**:
$(m\times k)(k\times n)=(m\times n)$ — mismatched shapes are the #1 bug in deep-learning
code. It is **not commutative** ($AB\ne BA$ in general). A batched forward pass takes a
data matrix $X$ (one sample per row, shape $N\times d$) and weights $W$ (shape $h\times d$)
to outputs $XW^\top$ (shape $N\times h$) — every sample through every neuron in one call.

## 7. Matrices as linear transformations

A matrix $W$ is a **linear transformation**: $W\mathbf x$ rotates, scales, and shears
$\mathbf x$. A layer $W\mathbf x + \mathbf b$ is an **affine** transformation (linear + a
shift), and **matrix multiplication composes transformations** — stacking layers *is*
multiplying their matrices. The **columns of $W$ are exactly where the basis vectors
$\mathbf e_1,\mathbf e_2$ land**, so the whole unit square maps to a parallelogram:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 230" style="width:100%;max-width:480px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A matrix is a linear transformation: the unit square with basis vectors e1 and e2 on the left becomes a sheared, scaled parallelogram on the right, whose edges are the columns of W — the images of e1 and e2."><defs><marker id="x7b" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#185FA5"/></marker><marker id="x7p" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#C0507A"/></marker></defs><rect x="1" y="1" width="478" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><polygon points="70,180 140,180 140,110 70,110" fill="#E6F1FB" stroke="#C9C8C1"/><line x1="70" y1="180" x2="136" y2="180" stroke="#185FA5" stroke-width="2.5" marker-end="url(#x7b)"/><line x1="70" y1="180" x2="70" y2="114" stroke="#C0507A" stroke-width="2.5" marker-end="url(#x7p)"/><text x="120" y="198" font-family="sans-serif" font-size="12" fill="#185FA5">e₁</text><text x="48" y="130" font-family="sans-serif" font-size="12" fill="#C0507A">e₂</text><text x="105" y="92" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">unit square</text><text x="240" y="140" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#33312E">apply W →</text><polygon points="320,180 404,159 439,89 355,110" fill="#FBEAD6" stroke="#C9C8C1"/><line x1="320" y1="180" x2="400" y2="160" stroke="#185FA5" stroke-width="2.5" marker-end="url(#x7b)"/><line x1="320" y1="180" x2="356" y2="112" stroke="#C0507A" stroke-width="2.5" marker-end="url(#x7p)"/><text x="406" y="174" font-family="sans-serif" font-size="12" fill="#185FA5">W e₁</text><text x="356" y="104" font-family="sans-serif" font-size="12" fill="#C0507A">W e₂</text><text x="392" y="92" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">transformed</text></svg></div>

The **Matrix-transform playground** (first tab) lets you drag the columns of $W$ and watch
the square deform live.

## 8. Special matrices & the transpose

- **Identity $I$** — leaves every vector unchanged ($I\mathbf x=\mathbf x$).
- **Diagonal** — scales each axis independently (no mixing).
- **Orthogonal $Q$** ($Q^\top Q=I$) — a pure **rotation/reflection**: it preserves lengths and angles.
- **Symmetric $S=S^\top$** — the well-behaved kind (real eigenvalues, orthogonal eigenvectors; §13).

The **transpose** $W^\top$ flips rows and columns and obeys $(AB)^\top=B^\top A^\top$. It is
not cosmetic: in backprop, if the forward pass is $\mathbf y=W\mathbf x$, the gradient flows
**backward through $W^\top$** — $\partial L/\partial\mathbf x = W^\top(\partial L/\partial\mathbf y)$.
Every framework's `.T` in a backward pass is this.

## 9. Span, independence, basis & rank

- **Span** — all vectors you can build as weighted sums of a set.
- **Linearly independent** — no vector in the set is a combination of the others (none is redundant).
- **Basis** — an independent set that spans the whole space (e.g. $\mathbf e_1,\mathbf e_2$ for the plane).
- **Rank** — the number of *independent directions* a matrix actually uses = the dimension
  of its output space. A **full-rank** matrix loses nothing; a **rank-deficient** one
  collapses dimensions.

In ML, rank is capacity: a $d\to h$ layer can pass at most $\min(d,h)$ independent
directions. **Bottleneck** layers and **LoRA** (post-training) deliberately use a *low*
rank $r\ll d$ to compress or to fine-tune cheaply.

## 10. The determinant & the inverse

The **determinant** $\det W$ is the signed factor by which $W$ scales area (2-D) or volume
(n-D). $\det W=0$ means $W$ **flattens** a dimension — it is **singular** and has no
inverse. When $\det W\ne0$, the **inverse** $W^{-1}$ undoes it ($W^{-1}W=I$). The normal
equations of linear regression, $\boldsymbol\beta=(X^\top X)^{-1}X^\top\mathbf y$ (M1), use
an inverse in *theory* — but in practice we **solve the system** rather than invert (faster
and numerically safer; see X6).

## 11. Orthogonality & projection — least squares is a projection

Project a vector onto a subspace = find the **closest point in it**; the leftover
(**residual**) is perpendicular to the subspace. This is exactly linear regression:
$\hat{\mathbf y}=X\boldsymbol\beta$ is the **projection of $\mathbf y$ onto the column
space of $X$**, so the residual $\mathbf y-\hat{\mathbf y}$ is orthogonal to every feature
— which *is* the normal equations. The "least-squares = orthogonal projection" claim from
M1 is this section.

## 12. Norms in depth

- $\lVert\mathbf x\rVert_2=\sqrt{\sum_i x_i^2}$ (Euclidean) → **ridge / weight decay**, and the distance metric everywhere.
- $\lVert\mathbf x\rVert_1=\sum_i|x_i|$ (Manhattan) → **lasso / sparsity**.
- $\lVert\mathbf x\rVert_\infty=\max_i|x_i|$ (largest component).
- **Frobenius** $\lVert W\rVert_F=\sqrt{\sum_{ij}W_{ij}^2}$ — the L2 norm of a matrix, used to penalize weights.

Why L1 gives *exact zeros*: its unit "ball" is a diamond with corners on the axes, so the
constrained optimum tends to land on a corner (a zero coordinate) — the geometry behind
lasso's feature selection.

## 13. Eigenvectors & eigenvalues

Some directions are special: an **eigenvector** $\mathbf v$ of a square matrix is only
**scaled**, not rotated — $W\mathbf v=\lambda\mathbf v$, with **eigenvalue** $\lambda$
its stretch factor. A generic vector, by contrast, gets rotated to a new direction:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 260 220" style="width:100%;max-width:300px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="An eigenvector v lies on a dashed line; applying W only stretches it to lambda v along the same line, while a non-eigenvector u is rotated to a new direction Wu."><defs><marker id="egA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#185FA5"/></marker><marker id="egB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#C0507A"/></marker></defs><rect x="1" y="1" width="258" height="218" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="13" y1="203" x2="236" y2="59" stroke="#9C9B95" stroke-width="1.3" stroke-dasharray="5 4"/><line x1="95" y1="150" x2="193" y2="86" stroke="#185FA5" stroke-width="2.6" marker-end="url(#egA)"/><line x1="95" y1="150" x2="154" y2="112" stroke="#185FA5" stroke-width="2.6" marker-end="url(#egA)"/><line x1="95" y1="150" x2="60" y2="89" stroke="#C0507A" stroke-width="2.6" marker-end="url(#egB)"/><line x1="95" y1="150" x2="78" y2="113" stroke="#C0507A" stroke-width="2.6" marker-end="url(#egB)"/><circle cx="95" cy="150" r="3" fill="#33312E"/><g font-family="sans-serif" font-size="12"><text x="150" y="126" fill="#185FA5">v</text><text x="150" y="80" fill="#185FA5">Wv = λv</text><text x="40" y="86" fill="#C0507A">u</text><text x="58" y="122" fill="#C0507A">Wu</text><text x="150" y="50" fill="#9C9B95" font-size="10">eigen-direction</text></g></svg></div>

**Symmetric** matrices are the friendly case (spectral theorem): real eigenvalues and
orthogonal eigenvectors. Two eigen-payoffs power the rest of the lab:

- **PCA (M5):** the top eigenvectors of the data's **covariance matrix** are the directions of most variance.
- **Optimization (X4):** the eigenvalues of the loss's curvature (Hessian) set how the bowl
  is shaped. A big spread — the **condition number** $\lambda_{\max}/\lambda_{\min}$ — is the
  **ravine** that makes gradient descent zig-zag; feature scaling evens the eigenvalues out.

## 14. The singular value decomposition (SVD)

Every matrix factors as $X=U\Sigma V^\top$: **rotate** ($V^\top$), **scale** each axis by a
**singular value** $\sigma_i\ge0$ ($\Sigma$), **rotate** again ($U$). The singular values
rank directions by importance, so keeping the **top $k$** gives the best low-rank
approximation — the math behind **compression, denoising, PCA, and LoRA** (a fine-tuning
update forced to be a small $U\Sigma V^\top$). PCA is just the SVD of the centered data —
the principal axes are the directions of maximum spread:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 470 210" style="width:100%;max-width:470px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A cloud of data points forms an elongated blob; PCA finds the principal axes — a long arrow PC1 along the direction of most variance and a short perpendicular arrow PC2."><defs><marker id="pcaA0" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#0E5E45"/></marker><marker id="pcaA1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="468" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><circle cx="232" cy="134" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="207" cy="117" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="195" cy="119" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="243" cy="153" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="195" cy="127" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="263" cy="124" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="230" cy="107" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="233" cy="142" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="144" cy="149" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="104" cy="145" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="116" cy="164" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="154" cy="162" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="238" cy="121" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="228" cy="131" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="133" cy="153" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="164" cy="134" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="289" cy="88" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="234" cy="146" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="193" cy="139" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="237" cy="127" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="156" cy="157" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="302" cy="67" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="283" cy="111" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="205" cy="182" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="268" cy="87" r="3.3" fill="#5B8FC2" opacity="0.82"/> <circle cx="239" cy="138" r="3.3" fill="#5B8FC2" opacity="0.82"/><line x1="230" y1="128" x2="363" y2="79" stroke="#0E5E45" stroke-width="2.6" marker-end="url(#pcaA0)"/><line x1="230" y1="128" x2="240" y2="156" stroke="#9A6A2A" stroke-width="2.6" marker-end="url(#pcaA1)"/><g font-family="sans-serif" font-size="11"><text x="300" y="66" fill="#0E5E45">PC1 — most variance</text><text x="246" y="176" fill="#9A6A2A">PC2</text><text x="120" y="30" fill="#6B6A66">each dot = a data sample</text></g></svg></div>

## 15. A worked forward pass (numbers)

Everything above meets in one line of a network. With
$W=\begin{bmatrix}1&-1\\2&0.5\end{bmatrix}$, $\mathbf x=\begin{bmatrix}1\\2\end{bmatrix}$,
$\mathbf b=\begin{bmatrix}0\\-1\end{bmatrix}$:

$$ W\mathbf x=\begin{bmatrix}1\cdot1+(-1)\cdot2\\2\cdot1+0.5\cdot2\end{bmatrix}=\begin{bmatrix}-1\\3\end{bmatrix},\quad \mathbf z=W\mathbf x+\mathbf b=\begin{bmatrix}-1\\2\end{bmatrix},\quad \mathrm{ReLU}(\mathbf z)=\begin{bmatrix}0\\2\end{bmatrix}. $$

Each output row is one neuron's dot product (§4) plus its bias, then a nonlinearity. Stack
another such layer and you are multiplying matrices (§7) — that is the whole forward pass.

## 16. Shapes & broadcasting (the practical bit)

In code you spend most of your debugging on **shapes**. A batch $X$ of shape $N\times d$
times weights $W^\top$ of shape $d\times h$ gives $N\times h$; the bias, shape $(h,)$,
**broadcasts** — the same bias vector is added to all $N$ rows. Mismatched inner dimensions
are the classic error. Rule of thumb: **write the shape of every tensor next to it**, and
matmuls chain as $(a\times b)(b\times c)=(a\times c)$.

## 17. The big picture — linear algebra runs the whole lab

| you'll see it as… | it's really… |
|---|---|
| a neuron $z=\mathbf w\cdot\mathbf x+b$ | a **dot product** |
| a layer / attention $QK^\top$ | a **matrix multiply** |
| gradients flowing back | multiplying by $W^\top$ |
| PCA / dimensionality reduction (M5) | **eigenvectors** of covariance / **SVD** |
| GD zig-zagging in a ravine (X4) | a bad **condition number** (eigenvalue spread) |
| LoRA fine-tuning | a **low-rank** $U\Sigma V^\top$ update |

**What to remember:** a dot product measures alignment; a matrix is a transformation and a
stack of dot products; eigen/SVD reveal the directions that matter. Everything else in the
lab is these, repeated.
"""

_TASKS = r"""
### In the Dot-product playground
1. Rotate $\mathbf x$ until it points the **same way** as $\mathbf w$ — what happens
   to the dot product and to $\theta$? Now make them **perpendicular**.
2. Point $\mathbf x$ **opposite** to $\mathbf w$ — what sign is the dot product?
3. Double the length of $\mathbf x$ — does $\cos\theta$ change? Does $\mathbf w\cdot\mathbf x$?

### In the Matrix-transform playground
4. Drag the columns of $W$ to make a **pure rotation** (the two column arrows stay unit-length
   and perpendicular) — does the area of the square change? Now make the two columns
   **parallel**: the square collapses to a line — what is $\det W$ then?
5. Turn on **eigenvectors**: find a $W$ where the test vector lands on its **own line**
   (an eigenvector) versus one where it is clearly **rotated**.

### Pencil & paper
6. Compute $\lVert(3,4)\rVert$, then $\cos\theta$ between $(1,0)$ and $(1,1)$.
7. Compute $(1,2,3)\cdot(0,1,0)$ and explain why it "selects" the 2nd component.
8. Multiply $\begin{bmatrix}1&-1\\2&0.5\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}$ by hand;
   state the output shape and check it against §15.
9. Read the eigenvectors and eigenvalues of the diagonal matrix $\begin{bmatrix}2&0\\0&3\end{bmatrix}$.
10. Explain in one line why a matrix with $\det W=0$ has no inverse.
11. Show that least squares $\hat{\mathbf y}=X\boldsymbol\beta$ makes the residual
    $\mathbf y-\hat{\mathbf y}$ orthogonal to every column of $X$.

### Code
12. Implement `dot`, `norm`, and `cosine_similarity` in NumPy; check against `np.dot` / `np.linalg.norm`.
13. Verify $\mathbf w\cdot\mathbf x=\lVert\mathbf w\rVert\lVert\mathbf x\rVert\cos\theta$ numerically for random vectors.
14. Take a 2-D point cloud, center it, run `np.linalg.svd`, and plot the top principal axis — reproduce the PCA figure.

### Bridge to ANN / ML
15. A neuron computes $z=\mathbf w\cdot\mathbf x+b$; a layer computes $W\mathbf x+\mathbf b$.
    Confirm the Playground neuron's $z$ equals your hand dot product plus the bias, and that
    stacking two layers = composing (multiplying) their matrices.
"""

_REFERENCES = r"""
### Books
- Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* — [free PDF](https://mml-book.github.io/) (Ch. 2–4 map exactly onto this page).
- Strang — *Introduction to Linear Algebra* (the classic; his MIT 18.06 lectures are free).

### Video & interactive
- **3Blue1Brown** — [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra): vectors & dot products, matrices as transformations, determinants, **eigenvectors**, and change of basis — the perfect companion to §7, §10, §13.
- Khan Academy — vectors, matrices, eigenvalues.

### In this lab
- ANN: the neuron's $z=\mathbf w\cdot\mathbf x+b$, a layer's $W\mathbf x+\mathbf b$, backprop through $W^\top$, attention's $QK^\top$.
- ML: cosine similarity & norms (search, regularization); **eigen/SVD** behind **PCA** (M5); the eigenvalue **condition number** behind the ravine in **X4 / Optimizers**.
"""

_QUIZ = [
    Question(
        "The geometric form of the dot product a·b is…",
        ["|a| + |b|", "|a||b|cos θ", "|a||b|sin θ", "a + b"],
        1,
        "a·b = |a||b|cos θ = Σ aᵢbᵢ. It measures alignment.",
    ),
    Question(
        "If a·b = 0 (and neither is the zero vector), the vectors are…",
        ["parallel", "perpendicular (orthogonal)", "equal", "opposite"],
        1,
        "cos θ = 0 means θ = 90°, so the vectors are orthogonal.",
    ),
    Question(
        "Cosine similarity ignores which property of the vectors?",
        ["their direction", "their magnitudes (length)", "their sign", "their dimension"],
        1,
        "Dividing by the norms removes length — only the direction (angle) matters.",
    ),
    Question(
        "The L2 norm of the vector (3, 4) is…",
        ["7", "5", "12", "1"],
        1,
        "√(3² + 4²) = √25 = 5.",
    ),
    Question(
        "A neuron's pre-activation z = w·x + b uses which operation between w and x?",
        ["element-wise max", "the dot product", "the cross product", "concatenation"],
        1,
        "z is the dot product of weights and inputs, plus the bias.",
    ),
    Question(
        "To multiply matrices of shapes (m×k) and (p×n), you need…",
        ["m = p", "k = p", "k = n", "m = n"],
        1,
        "The inner dimensions must match: (m×k)(k×n) = (m×n).",
    ),
    Question(
        "A layer computing Wx (W has one row per neuron) is doing…",
        ["a single dot product", "many dot products at once (one per row/neuron)",
         "a sort", "an activation"],
        1,
        "Matrix-vector multiply = a batch of dot products, one per neuron — the core deep-learning op.",
    ),
    Question(
        "In backprop, the gradient of the loss w.r.t. a layer's input flows through…",
        ["W", "Wᵀ (the transpose)", "W⁻¹", "the identity"],
        1,
        "Forward y = Wx; backward ∂L/∂x = Wᵀ(∂L/∂y). The transpose routes gradients back.",
    ),
    Question(
        "An eigenvector v of W satisfies…",
        ["Wv = 0", "Wv = λv (only scaled, not rotated)", "Wᵀv = v", "v·W = 1"],
        1,
        "Wv = λv: an eigenvector keeps its direction; λ is how much it is stretched.",
    ),
    Question(
        "If det(W) = 0, then W…",
        ["is orthogonal", "collapses a dimension and has no inverse", "is the identity",
         "has all eigenvalues equal to 1"],
        1,
        "A zero determinant means W flattens space (singular) — it can't be inverted.",
    ),
    Question(
        "PCA finds its principal axes from the top…",
        ["rows of the data", "eigenvectors of the covariance matrix (equivalently the SVD)",
         "largest data points", "biases"],
        1,
        "The top eigenvectors of the covariance (or the top singular vectors) are the directions of most variance.",
    ),
    Question(
        "Which norm encourages sparse solutions (many exact zeros)?",
        ["L2 (Euclidean)", "L1 (Manhattan)", "L∞", "the dot product"],
        1,
        "L1 (lasso) drives weights to exactly zero; L2 (ridge) shrinks but rarely zeroes them.",
    ),
    Question(
        "A large condition number (spread of eigenvalues) of the loss curvature causes…",
        ["faster convergence", "gradient descent to zig-zag down a ravine",
         "the model to overfit", "vanishing gradients"],
        1,
        "Uneven curvature (a ravine) makes GD zig-zag — the X4/Optimizers story; scaling evens it out.",
    ),
    Question(
        "Compute (1, 2, 3) · (0, 1, 0).",
        ["0", "2", "6", "5"],
        1,
        "1·0 + 2·1 + 3·0 = 2 — the dot product 'selects' the 2nd component.",
    ),
]

VECTORS = Lesson(
    key="vectors_dot",
    title="Vectors & the dot product",
    theory=_THEORY,
    quiz=_QUIZ,
    tasks=_TASKS,
    references=_REFERENCES,
)

REGISTRY = {VECTORS.key: VECTORS}
