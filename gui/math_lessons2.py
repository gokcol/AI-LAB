"""Math module lessons X2–X6. Reuses the Lesson/Question/quiz engine from lessons.py."""

from __future__ import annotations

from lessons import Lesson, Question

# ===========================================================================
# X2 — Calculus & gradients
# ===========================================================================

_X2_THEORY = r"""
## 1. Why calculus for ML

Training a model = **optimization**: find the parameters that minimize a loss. The tool
for "which way is downhill?" is the **derivative**. Every learning algorithm in this lab
— linear/logistic regression, SVMs, neural nets — moves parameters in the direction the
derivative points. Calculus is the engine; gradient descent (X4) is the car.

## 2. The derivative — rate of change

The derivative $f'(x)=\dfrac{df}{dx}$ is the **instantaneous rate of change** of $f$ —
the slope of its tangent line:
$$ f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}. $$
Example: $f(x)=x^2 \Rightarrow f'(x)=2x$, so at $x=3$ the function rises $6$ units per
unit of $x$. **Sign tells you direction** (uphill/downhill); **magnitude** tells you how
steep. At a minimum, $f'(x)=0$ (flat). The derivative *is* the slope of the line that just
grazes the curve at that point — its **rise over run**:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 420 260" style="width:100%;max-width:420px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A curve f(x) with the tangent line at a point P; the tangent's slope, rise over run, is the derivative f-prime at P."><rect x="1" y="1" width="418" height="258" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="40" y1="235" x2="408" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><line x1="40" y1="40" x2="40" y2="235" stroke="#C9C8C1" stroke-width="1.2"/><polyline points="40,123 49,133 58,143 68,152 77,160 86,168 95,175 105,181 114,187 123,192 132,196 142,200 151,204 160,206 169,208 178,210 188,210 197,211 206,210 215,209 225,207 234,205 243,202 252,198 262,194 271,189 280,184 289,177 298,171 308,163 317,155 326,147 335,137 345,127 354,117 363,106 372,94 382,81 391,68 400,55" fill="none" stroke="#185FA5" stroke-width="2.5"/><line x1="266" y1="216" x2="400" y2="64" stroke="#9A6A2A" stroke-width="2.2"/><line x1="349" y1="123" x2="379" y2="123" stroke="#9C9B95" stroke-width="1.3" stroke-dasharray="4 3"/><line x1="379" y1="123" x2="379" y2="89" stroke="#9C9B95" stroke-width="1.3" stroke-dasharray="4 3"/><circle cx="349" cy="123" r="4" fill="#33312E"/><g font-family="sans-serif" font-size="12"><text x="386" y="58" fill="#185FA5">f(x)</text><text x="252" y="84" fill="#9A6A2A">tangent: slope = f′(x)</text><text x="336" y="118" fill="#33312E">P</text><text x="358" y="137" font-size="10" fill="#9C9B95">Δx</text><text x="383" y="108" font-size="10" fill="#9C9B95">Δy</text></g></svg></div>

## 3. The rules you actually use

- **Power:** $\frac{d}{dx}x^n = n x^{n-1}$.
- **Sum:** the derivative of a sum is the sum of derivatives.
- **Product:** $(uv)' = u'v + uv'$.
- **Chain rule (the big one):** for a composition $f(g(x))$,
  $$ \frac{df}{dx} = f'(g(x))\cdot g'(x). $$
  Example: $\frac{d}{dx}\sin(x^2) = \cos(x^2)\cdot 2x$. **Backpropagation is just the
  chain rule applied over and over** through a network — derivatives multiply along the
  path (ANN §6).

## 4. Partial derivatives

Loss functions depend on *many* parameters. A **partial derivative** $\dfrac{\partial f}{\partial x_i}$ is the derivative w.r.t. one variable, **holding the others fixed**.
Example: $f(x,y)=x^2+3xy \Rightarrow \frac{\partial f}{\partial x}=2x+3y$,
$\frac{\partial f}{\partial y}=3x$.

## 5. The gradient — the steepest-ascent vector

Stack all partials into a vector — the **gradient**:
$$ \nabla f = \Big(\tfrac{\partial f}{\partial x_1},\dots,\tfrac{\partial f}{\partial x_n}\Big). $$
It points in the direction of **steepest ascent**; its negative $-\nabla f$ is steepest
**descent** — exactly the direction gradient descent steps in. Its magnitude is the
slope in that direction, and it is **perpendicular to the level sets** (contours) of $f$.
For $f(x,y)=x^2+y^2$, $\nabla f=(2x,2y)$ — at $(3,4)$ it points outward along $(6,8)$,
away from the minimum at the origin. On a contour map, $\nabla f$ always points **straight
across the contours, uphill**; gradient descent just walks the other way, $-\nabla f$:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 360 300" style="width:100%;max-width:360px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Concentric circular contours of a bowl with its minimum at the centre. At a point P the gradient points radially outward (steepest ascent), the negative gradient points inward toward the minimum (steepest descent), and both are perpendicular to the contour through P."><defs><marker id="x5up" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#A32D2D"/></marker><marker id="x5dn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#1D9E75"/></marker></defs><rect x="1" y="1" width="358" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g fill="none" stroke="#D9D8D1" stroke-width="1.4"><circle cx="180" cy="160" r="40"/><circle cx="180" cy="160" r="80"/><circle cx="180" cy="160" r="120"/></g><line x1="214" y1="80" x2="260" y2="126" stroke="#9C9B95" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="237" y1="103" x2="269" y2="71" stroke="#A32D2D" stroke-width="2.5" marker-end="url(#x5up)"/><line x1="237" y1="103" x2="205" y2="135" stroke="#1D9E75" stroke-width="2.5" marker-end="url(#x5dn)"/><circle cx="237" cy="103" r="4" fill="#33312E"/><circle cx="180" cy="160" r="3" fill="#33312E"/><g font-family="sans-serif" font-size="12"><text x="276" y="66" fill="#A32D2D">∇f (uphill)</text><text x="120" y="150" fill="#1D9E75">−∇f (descent)</text><text x="180" y="178" text-anchor="middle" font-size="10" fill="#6B6A66">min</text><text x="265" y="135" font-size="10" fill="#9C9B95">contour</text></g><text x="180" y="290" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#9C9B95">gradient ⟂ contours · descent walks toward the minimum</text></svg></div>

## 6. The multivariate chain rule = backprop

When functions compose ($\mathbf z = g(\mathbf x)$, $L = f(\mathbf z)$), gradients flow
**backward** by multiplying local derivatives:
$\dfrac{\partial L}{\partial x_i}=\sum_j \dfrac{\partial L}{\partial z_j}\dfrac{\partial z_j}{\partial x_i}$.
For vector-to-vector maps the table of partials is the **Jacobian** matrix. This chained,
backward multiplication is **precisely** what an autograd engine does (ANN e04/e06).

## 7. Curvature — second derivatives & the Hessian

The second derivative $f''$ measures **curvature** (how the slope changes). For many
variables it's the **Hessian** matrix of second partials. Curvature tells you whether a
flat point ($\nabla f=0$) is a minimum (curves up), maximum (curves down), or **saddle**
(both). It also governs how hard a function is to optimize — and underlies **convexity**
(X4): a convex loss curves up everywhere, so the one flat point is the global minimum.

## 8. Worked example — gradient of a tiny loss

Squared error for one point: $L(w)=(wx-y)^2$. By the chain rule,
$\dfrac{dL}{dw}=2(wx-y)\cdot x$. With $x=2,\ y=5,\ w=1$: prediction $wx=2$, error
$wx-y=-3$, so $\dfrac{dL}{dw}=2(-3)(2)=-12$. Negative ⇒ increasing $w$ lowers the loss ⇒
gradient descent moves $w$ **up**. (Generalize to all parameters and you have the
training update.)

## 9. A derivatives cheat-sheet

The handful you actually need:

| $f(x)$ | $f'(x)$ |
|---|---|
| $c$ (constant) | $0$ |
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ / $\cos x$ | $\cos x$ / $-\sin x$ |
| $\sigma(x)$ (sigmoid) | $\sigma(x)(1-\sigma(x))$ |
| $\max(0,x)$ (ReLU) | $0$ if $x<0$, else $1$ |

Combine with the **sum, product, quotient, and chain** rules and you can differentiate
*any* network — which is all an autograd engine does, automatically.

## 10. Linear approximation & Taylor

Near a point a smooth function looks **linear**:
$$ f(x+\Delta) \approx f(x) + f'(x)\,\Delta, \qquad f(\theta+\Delta)\approx f(\theta)+\nabla f\cdot\Delta. $$
This first-order **Taylor** view is *why* a small step $-\eta\nabla f$ lowers the loss —
it's the best local linear prediction. Add the curvature term $\tfrac12\Delta^\top H\Delta$
and you get the second-order picture behind Newton's method (X4).

## 11. Directional derivatives — why the gradient is "steepest"

The rate of change of $f$ as you move in a unit direction $\hat{\mathbf u}$ is
$\nabla f\cdot\hat{\mathbf u}$. By Cauchy–Schwarz that dot product is **largest when
$\hat{\mathbf u}$ points along $\nabla f$** — so $\nabla f$ is the direction of steepest
ascent and $-\nabla f$ steepest descent. The gradient packs every directional slope into
one vector.

## 12. Worked example — the chain rule through two layers

A tiny net: $z_1=w_1 x,\ a=\sigma(z_1),\ z_2=w_2 a,\ L=\tfrac12(z_2-y)^2$. Backprop just
multiplies local derivatives **backward**:
$$ \frac{\partial L}{\partial z_2}=z_2-y,\quad \frac{\partial L}{\partial w_2}=(z_2-y)\,a,\quad \frac{\partial L}{\partial a}=(z_2-y)\,w_2, $$
$$ \frac{\partial L}{\partial z_1}=\frac{\partial L}{\partial a}\,\sigma'(z_1),\quad \frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial z_1}\,x. $$
Each arrow is one chain-rule factor — that's the **entire backprop algorithm** in
miniature (ANN e04/e06).

## 13. Integrals — the reverse direction (brief)

Integration inverts differentiation (the Fundamental Theorem of Calculus) and measures
accumulated area. ML leans far more on derivatives, but integrals appear in probability:
an **expectation** is an integral $\int x\,p(x)\,dx$ (X3), and a valid distribution must
integrate to 1.

## 14. Limits & continuity

The derivative rests on the **limit**: $\lim_{x\to a}f(x)=L$ means $f$ gets arbitrarily
close to $L$ near $a$. $f$ is **continuous** at $a$ if $\lim_{x\to a}f(x)=f(a)$ (no
jumps). **Differentiable ⇒ continuous**, but not the reverse: $|x|$ is continuous at 0
yet has a **kink** (no derivative) there — exactly why ReLU's derivative is undefined at
0 and we use a *subgradient* (X4 §15).

## 15. Critical points & optimality conditions

At a max/min the gradient vanishes — the **first-order condition** $\nabla f=\mathbf 0$
(a *critical point*). Classify it by curvature, the **second-order condition** on the
**Hessian** $H$: positive-definite (all eigenvalues $>0$) → local **min**;
negative-definite → **max**; mixed signs → **saddle**. For a **convex** $f$, $\nabla f=\mathbf 0$ already guarantees a *global* min — why convex losses (X4) are easy.

## 16. The gradient cookbook (vector/matrix calculus)

The identities you reuse constantly ($\mathbf a, A$ constant):
- $\nabla_{\mathbf x}(\mathbf a^\top\mathbf x)=\mathbf a$
- $\nabla_{\mathbf x}(\mathbf x^\top\mathbf x)=2\mathbf x$, and $\nabla_{\mathbf x}\lVert\mathbf x\rVert=\mathbf x/\lVert\mathbf x\rVert$
- $\nabla_{\mathbf x}(\mathbf x^\top A\mathbf x)=(A+A^\top)\mathbf x$ ($=2A\mathbf x$ if $A$ symmetric)
- $\partial(W\mathbf x)/\partial\mathbf x = W$ — a layer's Jacobian **is** its weight matrix
- **MSE:** $\nabla_{\mathbf w}\tfrac1m\lVert X\mathbf w-\mathbf y\rVert^2=\tfrac2m X^\top(X\mathbf w-\mathbf y)$ (M1 §4)
- **softmax+cross-entropy:** $\partial L/\partial \mathbf z=\mathbf p-\mathbf y$ (X5 §15, M2, ANN e08)

These six cover most gradients you'll derive in the lab.

## 17. The Jacobian — derivatives of vector functions

When $f:\mathbb R^n\to\mathbb R^m$, its derivative is the **Jacobian** $J$, the $m\times n$ matrix $J_{ij}=\partial y_i/\partial x_j$. The multivariate chain rule is **Jacobian
multiplication**: for $\mathbf z=g(f(\mathbf x))$, $J_{\mathbf z/\mathbf x}=J_{\mathbf z/\mathbf y}\,J_{\mathbf y/\mathbf x}$. A neural net is a chain of such maps, and backprop
multiplies their Jacobians right-to-left.

## 18. Forward vs. reverse mode (why backprop is "reverse")

You can apply the chain rule **forward** (inputs→output) or **reverse** (output→inputs).
For $\mathbb R^n\to\mathbb R^1$ — *many parameters, one scalar loss*, i.e. ML — **reverse
mode** computes **all $n$ partials in a single backward pass** at the cost of ~one forward
pass; forward mode would need $n$ passes. That asymmetry is *why* **backprop =
reverse-mode autodiff** powers training (ANN e04).

## 19. Numerical differentiation

Approximate a derivative without algebra: **forward difference**
$\frac{f(x+h)-f(x)}{h}$ (error $O(h)$) or the better **central difference**
$\frac{f(x+h)-f(x-h)}{2h}$ (error $O(h^2)$). Used for **gradient checking** (verify
backprop, ANN e06) — but too slow to *train* with (one perturbation per parameter). The
X2 playground is exactly this, watched as $h\to0$.

## 20. Connection to ML & ANN

Every loss in the lab is minimized by computing its **gradient** and stepping downhill:
$\theta \leftarrow \theta - \eta\,\nabla_\theta L$ (X4). For deep nets, the gradient is
computed by **backprop** = the chain rule (§3, §12, §17–18) run over the computation
graph. Master the chain rule and the gradient, and the rest of deep-learning training is
bookkeeping.
"""

_X2_TASKS = r"""
### Pencil & paper
1. Differentiate: $3x^4$, $\sin(x)$, $e^{2x}$, and $\frac{d}{dx}(5x^2+2x-1)$.
2. Chain rule: $\frac{d}{dx}(3x+1)^5$ and $\frac{d}{dx}e^{-x^2}$.
3. For $f(x,y)=x^2y+y^3$, find $\partial f/\partial x$ and $\partial f/\partial y$, then $\nabla f$ at $(1,2)$.
4. Show that the gradient of $f(x,y)=x^2+y^2$ at any point aims directly away from the origin.

### Code
5. Implement **numerical differentiation** $\frac{f(x+h)-f(x-h)}{2h}$; compare to the
   analytic derivative of $x^2$ (this is the gradient check from ANN e06).
6. Plot $f(x)=x^2$ and its tangent line at $x=2$.

### Bridge
7. Connect §6 to the ANN module: write how the chain rule turns into backprop through a
   2-layer network.
"""

_X2_REFS = r"""
### Video & books
- **3Blue1Brown** — [Essence of Calculus](https://www.3blue1brown.com/topics/calculus) (intuition for derivatives, chain rule).
- Deisenroth, Faisal & Ong — *Mathematics for ML*, ch. 5 (vector calculus) — [free PDF](https://mml-book.github.io/).
- Khan Academy — derivatives, partial derivatives, gradient.

### In this lab
- ANN: backprop (§6 here) = the chain rule; e04 autograd, e06 gradient checking.
- Math: X4 (gradient descent uses ∇f), X1 (vectors — the gradient is a vector).
"""

_X2_QUIZ = [
    Question("The derivative f'(x) measures…",
             ["the area under f", "the instantaneous rate of change (slope of the tangent)",
              "the average of f", "the maximum of f"], 1,
             "f'(x) is the slope of f at x — how fast it changes."),
    Question("By the chain rule, d/dx of sin(x²) is…",
             ["cos(x²)", "cos(x²)·2x", "2x·sin(x)", "−cos(2x)"], 1,
             "Outer derivative cos(x²) times inner derivative 2x."),
    Question("A partial derivative ∂f/∂x is computed by…",
             ["differentiating w.r.t. x while holding the other variables fixed",
              "summing over all variables", "integrating over x", "setting x=0"], 0,
             "Treat the other variables as constants and differentiate w.r.t. x."),
    Question("The gradient ∇f points in the direction of…",
             ["steepest descent", "steepest ascent (its negative is steepest descent)",
              "the nearest minimum", "constant f"], 1,
             "∇f points uphill (steepest ascent); gradient descent steps along −∇f."),
    Question("Backpropagation is essentially…",
             ["matrix inversion", "the chain rule applied through the network's computation graph",
              "random search", "integration"], 1,
             "Backprop multiplies local derivatives backward along paths — the chain rule at scale."),
    Question("At a minimum of a smooth function, the gradient is…",
             ["maximal", "zero", "negative", "undefined"], 1,
             "∇f = 0 at a flat point; curvature (Hessian) tells min vs. max vs. saddle."),
    Question("For L(w) = (wx − y)², dL/dw equals…",
             ["2(wx − y)", "2(wx − y)·x", "x²", "(wx − y)/x"], 1,
             "Chain rule: outer 2(wx−y) times inner derivative x."),
]

CALCULUS = Lesson("calculus", "Calculus & gradients", _X2_THEORY, _X2_QUIZ, _X2_TASKS, _X2_REFS)


# ===========================================================================
# X3 — Probability & statistics
# ===========================================================================

_X3_THEORY = r"""
## 1. Why probability for ML

Data is noisy, models are uncertain, and most **loss functions come straight from
probability** (maximum likelihood). Probability is the language for "how sure are we?"
— and classifiers literally output probabilities (M2).

## 2. Random variables & distributions

A **random variable** takes values with certain probabilities. **Discrete** ones use a
**PMF** $P(X=x)$ (probabilities sum to 1); **continuous** ones use a **PDF** $p(x)$
(integrates to 1, and $p(x)$ is a *density*, not a probability). A distribution is the
full description of those probabilities.

## 3. Key distributions — a catalog

A distribution is fixed by its **support** (possible values), a **PMF** (discrete) or
**PDF** (continuous), and its **parameters**. Here are the ones you'll actually meet, with
mean, variance, and where they appear in ML. A few of the **shapes** you'll learn to
recognize:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 480 175" style="width:100%;max-width:480px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Four distribution shapes side by side: a symmetric bell (normal), a flat box (uniform), a decaying curve (exponential), and a right-skewed hump with a long right tail."><rect x="1" y="1" width="478" height="173" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#D9D8D1" stroke-width="1"><line x1="20" y1="140" x2="120" y2="140"/><line x1="140" y1="140" x2="240" y2="140"/><line x1="260" y1="140" x2="360" y2="140"/><line x1="380" y1="140" x2="480" y2="140"/></g><polyline points="25,139 28,138 31,137 34,135 37,131 41,127 44,120 47,112 50,103 53,92 56,80 59,69 62,59 65,52 68,48 72,48 75,52 78,59 81,69 84,80 87,92 90,103 93,112 96,120 99,127 103,131 106,135 109,137 112,138 115,139" fill="none" stroke="#185FA5" stroke-width="2.4"/><polyline points="150,140 158,140 158,48 221,48 221,140 230,140" fill="none" stroke="#185FA5" stroke-width="2.4"/><polyline points="265,48 268,58 271,66 274,74 277,81 281,87 284,93 287,98 290,102 293,106 296,109 299,113 302,116 305,118 308,120 312,122 315,124 318,126 321,127 324,129 327,130 330,131 333,132 336,133 339,133 343,134 346,135 349,135 352,136 355,136" fill="none" stroke="#185FA5" stroke-width="2.4"/><polyline points="385,140 388,137 391,130 394,121 397,110 401,99 404,89 407,80 410,71 413,64 416,59 419,54 422,51 425,49 428,48 432,48 435,49 438,50 441,52 444,55 447,58 450,61 453,65 456,68 459,72 463,76 466,79 469,83 472,87 475,90" fill="none" stroke="#185FA5" stroke-width="2.4"/><g font-family="sans-serif" font-size="11" fill="#6B6A66" text-anchor="middle"><text x="70" y="162">Normal</text><text x="190" y="162">Uniform</text><text x="310" y="162">Exponential</text><text x="430" y="162">Right-skewed</text></g></svg></div>

### Discrete

| distribution | PMF | mean | variance | shows up as |
|---|---|---|---|---|
| **Bernoulli($p$)** | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | a single binary label / coin flip |
| **Binomial($n,p$)** | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | # successes in $n$ trials |
| **Categorical($\boldsymbol\pi$)** | $\pi_k$ | — | — | a multiclass label (the softmax target) |
| **Multinomial($n,\boldsymbol\pi$)** | counts coef. | $n\pi_k$ | $n\pi_k(1-\pi_k)$ | word counts (bag-of-words) |
| **Poisson($\lambda$)** | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | counts of rare events (mean = var) |
| **Geometric($p$)** | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | trials until the first success |

### Continuous

| distribution | PDF | mean | variance | shows up as |
|---|---|---|---|---|
| **Uniform($a,b$)** | $\tfrac{1}{b-a}$ | $\tfrac{a+b}{2}$ | $\tfrac{(b-a)^2}{12}$ | priors, weight init, sampling |
| **Gaussian($\mu,\sigma^2$)** | $\tfrac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | noise, init, the CLT — everywhere |
| **Exponential($\lambda$)** | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | waiting times; memoryless |
| **Gamma($\alpha,\beta$)** | $\propto x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | positive & skewed; sum of exponentials |
| **Beta($\alpha,\beta$)** | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\tfrac{\alpha}{\alpha+\beta}$ | — | a distribution *over* probabilities; Bernoulli's conjugate prior |
| **Laplace($\mu,b$)** | $\tfrac{1}{2b}e^{-|x-\mu|/b}$ | $\mu$ | $2b^2$ | heavy tails; the L1 / lasso prior |

### The Gaussian, special

It dominates because of the **CLT** (§16): sums/averages of many independent effects tend
to Gaussian. The **standard normal** $\mathcal N(0,1)$ is the $z$-standardized version
(§9); the **multivariate** Gaussian uses a mean *vector* and a covariance *matrix*; and
squared-error loss is the MLE under Gaussian noise (§12, M1).

### How they relate (a small map)

- Sum of **Bernoulli** trials → **Binomial**; Binomial with large $n$, small $p$ →
  **Poisson**; Binomial with large $n$ → **Gaussian** (CLT).
- **Categorical** generalizes Bernoulli to $K$ outcomes; **Multinomial** generalizes Binomial.
- **Exponential** is a special **Gamma** ($\alpha=1$); a sum of exponentials is Gamma.
- **Beta** is the conjugate prior for Bernoulli/Binomial; its multivariate cousin
  **Dirichlet** is the conjugate prior for Categorical/Multinomial (§18).

### Choosing one to model with

Match the **support** to your data: binary → Bernoulli, counts → Poisson, a proportion in
$[0,1]$ → Beta, a positive amount → Gamma/Exponential, a real value → Gaussian, one of $K$
categories → Categorical. A model's **output layer mirrors this** — sigmoid (Bernoulli),
softmax (Categorical), linear (Gaussian).

### Where each shows up (in practice)

- **Bernoulli** — the workhorse of **binary outcomes**: spam/not, click/no-click,
  fraud/legit, churn/stay, a pixel on/off. It *is* the output of **logistic regression**
  and a **sigmoid neuron**, and the per-trial atom of an A/B test.
- **Binomial** — **aggregate counts** of binary trials: conversions out of $n$ visitors,
  defective items per batch, votes in a sample. Behind **proportion confidence intervals**
  and A/B-test significance.
- **Poisson** — counts of **rare, independent events** per fixed window: hits per minute,
  customer arrivals, photons on a sensor, typos per page, mutations per genome.
  **Poisson regression** models count targets; queueing & reliability rely on it.
- **Geometric** — the **wait (in trials) until the first success**: attempts until a user
  converts, retransmissions until a packet gets through. It is **memoryless** — past
  failures don't change what's left.
- **Gaussian** — by far the most used: **measurement noise**, **regression residuals**,
  features after **standardization**, **weight initialization**, the **CLT** limit of
  averages, and **VAE** latents. The default "assume normal" when you know only a mean and
  a spread.
- **Beta** — a distribution **over a probability** in $[0,1]$: an unknown click-through or
  conversion **rate**, the **prior** in Bayesian A/B testing, and the exploration engine
  of **Thompson-sampling bandits**; also Naive-Bayes smoothing.
- **Gamma** — **positive, skewed** amounts: time-to-failure, insurance **claim sizes**,
  rainfall totals, the wait for $k$ Poisson events. It's the **conjugate prior** for a
  Poisson rate and for a Gaussian's precision.

## 4. Expectation & variance

The **expectation** $\mathbb E[X]=\sum_x x\,P(x)$ (or $\int x\,p(x)\,dx$) is the
long-run average. **Variance** $\operatorname{Var}[X]=\mathbb E[(X-\mathbb E[X])^2]$
measures spread; **std** $=\sqrt{\operatorname{Var}}$ is in the same units. Expectation is
**linear**: $\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y]$ (always, even if dependent).

## 5. Joint, marginal, conditional & independence

$P(A,B)$ is the **joint**; summing out a variable gives the **marginal**; $P(A\mid B)= \frac{P(A,B)}{P(B)}$ is the **conditional** ("A given B"). $A,B$ are **independent** iff
$P(A,B)=P(A)P(B)$. Independence (often the *naive* assumption) makes models tractable.

## 6. Bayes' rule — updating beliefs

$$ P(H\mid D)=\frac{P(D\mid H)\,P(H)}{P(D)} \;\;\propto\;\; \underbrace{P(D\mid H)}_{\text{likelihood}}\,\underbrace{P(H)}_{\text{prior}}. $$
**Worked example (the base-rate trap).** A disease affects 1% of people. A test is 99%
accurate both ways. You test positive — chance you're sick? Of 10,000 people, 100 are
sick (99 test +), 9,900 healthy (99 false +). So $P(\text{sick}\mid +)=99/(99+99)= \mathbf{50\%}$, not 99%. Rare conditions + imperfect tests ⇒ many false positives — the
intuition behind precision/recall (M2).

## 7. Maximum likelihood estimation (MLE)

Choose parameters that make the observed data **most probable**: maximize the
**likelihood** $\prod_i p(x_i\mid\theta)$, or equivalently the **log-likelihood**
$\sum_i \log p(x_i\mid\theta)$. The punchline that ties the whole ML module together:
- Gaussian noise + MLE $\Rightarrow$ **minimize squared error** (M1).
- Bernoulli/categorical + MLE $\Rightarrow$ **minimize cross-entropy** (M2).

So "minimize the loss" is usually "**maximize the likelihood**" in disguise.

## 8. Covariance, correlation & sampling

**Covariance** measures how two variables move together; **correlation** normalizes it
to $[-1,1]$. We never see the true distribution — only **samples** — so we use
**estimators** (e.g. the sample mean), which have their own bias and variance. The **Law
of Large Numbers** says sample averages converge to $\mathbb E[X]$ as $n$ grows.

## 9. The Normal distribution in depth

The **bell curve**: symmetric around the mean $\mu$, with almost all its mass within
$3\sigma$ — the famous **68–95–99.7** rule made visible:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 460 285" style="width:100%;max-width:460px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The normal bell curve centred at the mean. About 68 percent of the area lies within one standard deviation, 95 percent within two, and 99.7 percent within three."><defs><marker id="x3ar" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#6B6A66"/></marker></defs><rect x="1" y="1" width="458" height="283" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="40" y1="225" x2="420" y2="225" stroke="#C9C8C1" stroke-width="1.2"/><g stroke="#D9D8D1" stroke-width="1.2" stroke-dasharray="4 3"><line x1="182" y1="121" x2="182" y2="225"/><line x1="278" y1="121" x2="278" y2="225"/><line x1="135" y1="202" x2="135" y2="225"/><line x1="325" y1="202" x2="325" y2="225"/><line x1="88" y1="223" x2="88" y2="225"/><line x1="372" y1="223" x2="372" y2="225"/></g><polyline points="40,225 53,225 65,225 78,224 91,223 103,220 116,215 129,207 141,195 154,177 167,155 179,128 192,101 205,77 217,60 224,56 230,54 236,56 243,60 255,77 268,101 281,128 293,155 306,177 319,195 331,207 344,215 357,220 369,223 382,224 395,225 407,225 420,225" fill="none" stroke="#185FA5" stroke-width="2.6"/><line x1="182" y1="238" x2="278" y2="238" stroke="#6B6A66" stroke-width="1.3" marker-start="url(#x3ar)" marker-end="url(#x3ar)"/><line x1="135" y1="254" x2="325" y2="254" stroke="#6B6A66" stroke-width="1.3" marker-start="url(#x3ar)" marker-end="url(#x3ar)"/><line x1="88" y1="269" x2="372" y2="269" stroke="#6B6A66" stroke-width="1.3" marker-start="url(#x3ar)" marker-end="url(#x3ar)"/><g font-family="sans-serif" font-size="11" fill="#33312E"><text x="284" y="242">68%</text><text x="331" y="258">95%</text><text x="378" y="273">99.7%</text></g><text x="230" y="221" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">μ</text></svg></div>

- **68–95–99.7 rule:** ~68% of the mass lies within $1\sigma$ of the mean, ~95% within
  $2\sigma$, ~99.7% within $3\sigma$.
- **z-score / standardization** $z=\frac{x-\mu}{\sigma}$ recenters data to mean 0, std 1
  — exactly the **feature scaling** of M7, and why "standardize your features" is
  everywhere.
- A **sum of independent Gaussians is Gaussian**; the **multivariate** Gaussian (a mean
  *vector* + a covariance *matrix*) generalizes the bell curve to many dimensions.

## 10. Variance algebra & standard error

$\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$, and for **independent** $X,Y$,
$\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)$. So the **sample
mean** of $n$ i.i.d. points has variance $\sigma^2/n$ and **standard error**
$\sigma/\sqrt n$ — the $1/\sqrt n$ shrinkage you watched in the CLT playground, and the
reason more data gives steadier estimates.

### The same two rules, applied to a neural network: where Xavier and He come from

Those two identities are enough to derive the initialisation schemes in `core/init.py`,
which otherwise look like magic constants. A unit computes $z=\sum_{i=1}^{n} w_i x_i$ over
$n=\text{fan\_in}$ inputs. Take the weights independent of the inputs, both zero-mean.
Then $\operatorname{Var}(w_ix_i)=\operatorname{Var}(w)\operatorname{Var}(x)$, and because
the terms are independent the second rule lets you simply add them up $n$ times:

$$ \operatorname{Var}(z)\;=\;n\,\operatorname{Var}(w)\,\operatorname{Var}(x) $$

That is the whole derivation. Now demand that a layer neither amplifies nor shrinks the
signal — $\operatorname{Var}(z)=\operatorname{Var}(x)$ — and solve:

$$ n\,\operatorname{Var}(w)=1 \quad\Longrightarrow\quad \operatorname{Var}(w)=\frac1n,\qquad \sigma_w=\sqrt{\frac{1}{n}} $$

**That is Xavier/Glorot init.** Not a tuned constant — the only scale that leaves the
variance unchanged. (Glorot's paper averages fan-in and fan-out, $2/(n_{\text{in}}+n_{\text{out}})$,
to balance the backward pass too; `core/init.py` uses the fan-in form.)

**Now add ReLU, and you lose exactly half.** For $z$ symmetric about zero, ReLU zeroes the
negative half and keeps the positive half untouched, so

$$ \mathbb E\big[\mathrm{relu}(z)^2\big] \;=\; \tfrac12\,\mathbb E\big[z^2\big] $$

*(measured: 0.50024 over 2 million samples).* Each layer therefore halves the variance —
compounding to $2^{-L}$ over $L$ layers. Compensate by doubling the weight variance:

$$ \operatorname{Var}(w)=\frac2n,\qquad \sigma_w=\sqrt{\frac{2}{n}} $$

**That is He/Kaiming init**, and the 2 is the ReLU factor, nothing else.

The prediction is sharp and testable: with ReLU and $\sqrt{1/n}$, the activation standard
deviation should fall by $1/\sqrt2\approx0.7071$ per layer. Measured over 12 layers of 256
units: **0.7067**. Four significant figures from three lines of algebra.

| layer | $\sigma_w=0.05$ (a "reasonable" guess) | $\sqrt{1/n}$ (Xavier) | $\sqrt{2/n}$ (He) |
|---|---|---|---|
| 1 | 4.7e-01 | 0.583 | 0.825 |
| 3 | 1.4e-01 | 0.273 | 0.773 |
| 6 | 2.7e-02 | 0.104 | 0.830 |
| 9 | 5.2e-03 | 0.039 | 0.880 |
| 12 | 9.8e-04 | 0.014 | 0.911 |

The middle column is the one to stare at: Xavier is *correct algebra for the wrong
nonlinearity*, and it still dies — a factor of 0.7 per layer is nearly invisible at depth 3
and fatal at depth 12. The left column is what picking a round number gets you. Only the
right column is flat, and it is flat because it was derived to be.

This is the concrete answer to "why did my deep network never start learning": the signal
was gone before it reached the loss, and the gradient — which obeys the same algebra
running backwards — was gone before it reached layer 1. Play with it in the **Playground**
tab, and see the same collapse from the gradient's side on the **Activations** page.

## 11. Probability vs. likelihood (a crucial distinction)

The same formula $p(x\mid\theta)$ reads two ways: as a function of **$x$** with $\theta$
fixed it's a **probability** (integrates to 1); as a function of **$\theta$** with the
data $x$ fixed it's the **likelihood** (it does *not* integrate to 1). MLE maximizes the
likelihood view — "which parameters make what I saw most plausible?"

## 12. Worked MLE — the Bernoulli

Flip a coin $n$ times, see $k$ heads. Likelihood $L(p)=p^k(1-p)^{n-k}$; log-likelihood
$\ell(p)=k\log p+(n-k)\log(1-p)$. Set $\ell'(p)=\frac{k}{p}-\frac{n-k}{1-p}=0$ and solve:
$$ \hat p = \frac{k}{n} \quad(\text{the sample mean}). $$
Intuitive — and the template for every MLE. (Logistic regression's MLE has no closed
form, so we use gradient descent, X4.)

## 13. Bayesian vs. frequentist — and where regularization comes from

Frequentist MLE picks the single best-fitting $\theta$. The **Bayesian** view keeps a
whole posterior $p(\theta\mid D)\propto p(D\mid\theta)\,p(\theta)$ (Bayes, §6); the
**MAP** estimate maximizes it. Beautifully, a **Gaussian prior** on the weights = **L2
regularization (ridge)**, and a **Laplace prior** = **L1 (lasso)**. Priors *are*
regularizers — the bridge between probability and M1/M6.

## 14. The exponential family & GLMs

Bernoulli, Gaussian, Poisson, Categorical, Beta, and Gamma (the §3 catalog) all belong to
the **exponential family** — distributions writable as
$p(x\mid\theta)\propto e^{\,\eta(\theta)\cdot T(x)}$. That shared structure is the backbone
of **generalized linear models (GLMs)**: choose the output distribution and its natural
link, and a model falls out — Gaussian → **linear regression**, Bernoulli → **logistic
regression**, Poisson → **Poisson regression**, Categorical → **softmax regression**. One
framework, many models — and each is a single neuron with the matching activation/loss.

## 15. Law of total probability & total expectation

- **Total probability:** $P(A)=\sum_b P(A\mid B{=}b)\,P(B{=}b)$ — average over cases.
- **Total expectation:** $\mathbb E[X]=\mathbb E\big[\,\mathbb E[X\mid Y]\,\big]$.
- Handy identity: $\operatorname{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2$.

Conditioning lets you break a hard probability into manageable cases.

## 16. Law of Large Numbers & CLT, precisely

- **LLN:** the sample mean $\bar X_n\to\mathbb E[X]$ as $n\to\infty$ — averages converge to the truth.
- **CLT:** for i.i.d. $X_i$ with mean $\mu$, variance $\sigma^2$, the standardized mean
  $\frac{\bar X_n-\mu}{\sigma/\sqrt n}\to\mathcal N(0,1)$, i.e. $\bar X_n\approx\mathcal N(\mu,\sigma^2/n)$ — the shrinking-spread Gaussian from the X3 playground.

## 17. Estimator quality — bias, variance, consistency

An **estimator** $\hat\theta$ is a function of the data, judged by:
- **bias** $=\mathbb E[\hat\theta]-\theta$ (systematic error),
- **variance** $=\operatorname{Var}(\hat\theta)$ (sensitivity to the sample),
- **MSE** $=\text{bias}^2+\text{variance}$ — the *same* decomposition as M0/M7,
- **consistency** — $\hat\theta\to\theta$ as $n\to\infty$.

MLE is consistent and asymptotically efficient under mild conditions.

## 18. Conjugate priors — Bayes, worked

A **conjugate** prior keeps the posterior in the same family. For a Bernoulli rate with a
**Beta($\alpha,\beta$)** prior, after $k$ heads in $n$ flips the posterior is
**Beta($\alpha+k,\ \beta+n-k$)** — you just add counts. Its mean
$\frac{\alpha+k}{\alpha+\beta+n}$ is a **smoothed** estimate (the prior acts as
pseudo-counts) — exactly **Laplace smoothing** in Naive Bayes. Bayesian updating =
accumulating evidence.

## 19. Sampling & Monte Carlo

We rarely integrate distributions by hand; we **sample**. **Monte Carlo** estimates an
expectation by averaging samples: $\mathbb E[f(X)]\approx\frac1N\sum_i f(x_i)$. It
underlies simulation, Bayesian inference (MCMC), dropout-as-ensemble, and RL return
estimates.

## 20. A word on inference — confidence & significance

A **confidence interval** quantifies estimate uncertainty (e.g. $\bar x\pm1.96\,\text{SE}$
for ~95%); a **hypothesis test** asks whether an effect could be chance (the **p-value**).
ML cares more about *prediction* than *inference*, but you meet these when comparing
models — and "**correlation ≠ causation**" is the eternal caveat.

## 21. Connection to ML & ANN

Losses are **negative log-likelihoods**; classifiers output **probability distributions**
(softmax); **Naive Bayes** applies Bayes' rule with an independence assumption;
calibration (M2 §12) asks whether predicted probabilities are honest; and the Gaussian
underlies weight initialization and noise models in neural nets.
"""

_X3_TASKS = r"""
### Pencil & paper
1. A fair die: compute E[X] and Var[X].
2. Redo the Bayes disease example with a 1-in-1000 base rate — now what is P(sick | +)?
3. Show that maximizing the Gaussian log-likelihood of residuals = minimizing squared error.
4. Two independent coin flips: write the joint distribution; verify independence.

### Code
5. Sample 10,000 draws from `np.random.normal`; plot the histogram and overlay the PDF.
6. Demonstrate the **Central Limit Theorem**: average k uniform samples for k = 1, 2, 30
   and watch the distribution of the average become Gaussian.
7. Implement Bayes' rule for the disease example and verify the 50% answer.

### Bridge
8. Connect §7 to M1/M2: MSE ↔ Gaussian MLE, cross-entropy ↔ Bernoulli/categorical MLE.
"""

_X3_REFS = r"""
### Books & video
- Deisenroth, Faisal & Ong — *Mathematics for ML*, ch. 6 (probability) — [free PDF](https://mml-book.github.io/).
- Blitzstein & Hwang — *Introduction to Probability* (excellent, free online).
- **3Blue1Brown** — Bayes' theorem & binomial videos. · StatQuest — distributions, likelihood.

### In this lab
- ML: M1 (MSE = Gaussian MLE), M2 (cross-entropy = Bernoulli MLE, calibration).
- Math: X5 (cross-entropy / KL), X4 (the optimization that maximizes likelihood).
"""

_X3_QUIZ = [
    Question("A probability density function (PDF) describes…",
             ["a discrete count", "a continuous random variable; it integrates to 1",
              "the mean only", "a single probability value"], 1,
             "Continuous variables use a PDF (density, integrates to 1); discrete use a PMF."),
    Question("The Gaussian is ubiquitous largely because of…",
             ["Bayes' rule", "the Central Limit Theorem (sums of many effects tend to Gaussian)",
              "the chain rule", "gradient descent"], 1,
             "Averages/sums of many independent effects converge to a Gaussian (CLT)."),
    Question("Expectation E[X] is…",
             ["the most likely value", "the long-run average value of X", "the variance", "always 0"], 1,
             "E[X] is the probability-weighted average; variance measures spread around it."),
    Question("Bayes' rule says the posterior is proportional to…",
             ["prior ÷ likelihood", "likelihood × prior", "evidence × posterior", "likelihood only"], 1,
             "P(H|D) ∝ P(D|H)·P(H): likelihood times prior."),
    Question("In the 1%-disease, 99%-accurate-test example, testing positive means P(sick) ≈…",
             ["99%", "50%", "1%", "90%"], 1,
             "Equal numbers of true and false positives → about 50%; the base rate dominates."),
    Question("Minimizing squared error corresponds to maximum likelihood under…",
             ["uniform noise", "Gaussian noise", "no noise", "Bernoulli labels"], 1,
             "Gaussian-noise MLE ⇒ least squares; Bernoulli/categorical MLE ⇒ cross-entropy."),
    Question("Two events are independent iff…",
             ["P(A,B) = P(A) + P(B)", "P(A,B) = P(A)·P(B)", "P(A|B) = P(B)", "they never co-occur"], 1,
             "Independence means the joint factorizes into the product of marginals."),
]

PROBABILITY = Lesson("probability", "Probability & statistics", _X3_THEORY, _X3_QUIZ, _X3_TASKS, _X3_REFS)


# ===========================================================================
# X4 — Optimization
# ===========================================================================

_X4_THEORY = r"""
## 1. The problem

Almost all ML training is **optimization**: find parameters $\theta$ that minimize a loss
$L(\theta)$. We rarely have a closed form (logistic regression, neural nets don't), so we
**iterate downhill** using the gradient (X2).

## 2. Gradient descent

Repeatedly step in the direction of steepest descent, $-\nabla L$:
$$ \theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta). $$
$\eta$ is the **learning rate** (step size). Geometrically you're a ball rolling downhill
on the loss surface. Watch it in the **Playground**: the path follows the negative
gradient, perpendicular to the contours.

## 3. The learning rate — the most important knob

- **Too large** → you overshoot the minimum and **diverge / oscillate** (the loss blows
  up). Try it in the Playground by cranking $\eta$.
- **Too small** → painfully **slow** convergence.
- In practice: tune it, and often **decay** it over time (learning-rate schedules,
  warmup) for fast-then-fine progress.

The same bowl, three step sizes — watch the dots descend (amber path = the iterates):

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 540 230" style="width:100%;max-width:540px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient descent on the same quadratic bowl with three learning rates. Too small: tiny steps that crawl. Just right: a few steps converge to the bottom. Too big: steps overshoot and bounce across the valley without settling."><rect x="1" y="1" width="538" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><polyline points="12,40 16,55 19,69 23,83 27,95 31,107 34,118 38,128 42,138 46,147 49,154 53,161 57,168 61,173 64,178 68,182 72,185 76,188 79,189 83,190 87,190 91,189 94,188 98,185 102,182 106,178 109,173 113,168 117,161 121,154 124,147 128,138 132,128 136,118 139,107 143,95 147,83 151,69 154,55 158,40" fill="none" stroke="#BFD8EF" stroke-width="2.2"/><polyline points="192,40 196,55 199,69 203,83 207,95 211,107 214,118 218,128 222,138 226,147 229,154 233,161 237,168 241,173 244,178 248,182 252,185 256,188 259,189 263,190 267,190 271,189 274,188 278,185 282,182 286,178 289,173 293,168 297,161 301,154 304,147 308,138 312,128 316,118 319,107 323,95 327,83 331,69 334,55 338,40" fill="none" stroke="#BFD8EF" stroke-width="2.2"/><polyline points="372,40 376,55 379,69 383,83 387,95 391,107 394,118 398,128 402,138 406,147 409,154 413,161 417,168 421,173 424,178 428,182 432,185 436,188 439,189 443,190 447,190 451,189 454,188 458,185 462,182 466,178 469,173 473,168 477,161 481,154 484,147 488,138 492,128 496,118 499,107 503,95 507,83 511,69 514,55 518,40" fill="none" stroke="#BFD8EF" stroke-width="2.2"/><polyline points="19,68 26,92 32,110 37,125 42,138 46,148 50,156 54,162" fill="none" stroke="#9A6A2A" stroke-width="1.4"/><polyline points="199,68 245,179 259,189 263,190 265,190" fill="none" stroke="#9A6A2A" stroke-width="1.4"/><polyline points="427,181 463,181 426,180 465,179 425,178 466,177 423,176 468,175" fill="none" stroke="#9A6A2A" stroke-width="1.4"/><g fill="#9A6A2A"><circle cx="19" cy="68" r="3.2"/><circle cx="26" cy="92" r="3.2"/><circle cx="32" cy="110" r="3.2"/><circle cx="37" cy="125" r="3.2"/><circle cx="42" cy="138" r="3.2"/><circle cx="46" cy="148" r="3.2"/><circle cx="50" cy="156" r="3.2"/><circle cx="54" cy="162" r="3.2"/><circle cx="199" cy="68" r="3.2"/><circle cx="245" cy="179" r="3.2"/><circle cx="259" cy="189" r="3.2"/><circle cx="263" cy="190" r="3.2"/><circle cx="265" cy="190" r="3.2"/><circle cx="427" cy="181" r="3.2"/><circle cx="463" cy="181" r="3.2"/><circle cx="426" cy="180" r="3.2"/><circle cx="465" cy="179" r="3.2"/><circle cx="425" cy="178" r="3.2"/><circle cx="466" cy="177" r="3.2"/><circle cx="423" cy="176" r="3.2"/><circle cx="468" cy="175" r="3.2"/></g><g font-family="sans-serif" font-size="11.5" text-anchor="middle" fill="#33312E"><text x="85" y="24">η too small</text><text x="265" y="24">η just right</text><text x="445" y="24">η too big</text></g><g font-family="sans-serif" font-size="10" text-anchor="middle" fill="#9C9B95"><text x="85" y="212">crawls — slow</text><text x="265" y="212">converges fast</text><text x="445" y="212">overshoots / oscillates</text></g></svg></div>

## 4. Convex vs. non-convex

- **Convex** loss (one bowl): every local minimum is the **global** minimum — linear
  regression, logistic regression, SVM. Gradient descent is guaranteed to get there.
- **Non-convex** (neural nets): many local minima, saddles, and plateaus. No guarantee,
  yet GD works remarkably well in practice (good-enough minima are everywhere in high
  dimensions).

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 520 230" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: a convex loss is a single bowl, so descent always reaches the global minimum. Right: a non-convex loss has several dips, and gradient descent can get stuck in a local minimum that is not the global one."><rect x="1" y="1" width="518" height="228" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><polyline points="60,45 66,65 73,83 79,100 86,115 92,129 99,142 105,154 112,164 118,172 124,179 131,185 137,190 144,193 150,195 157,195 163,194 169,192 176,188 182,183 189,176 195,168 202,159 208,148 215,136 221,123 227,108 234,92 240,74 247,55 250,45" fill="none" stroke="#185FA5" stroke-width="2.5"/><circle cx="155" cy="189" r="6" fill="#C0507A"/><polyline points="290,119 296,103 303,84 309,67 316,58 322,60 329,73 335,97 342,126 348,155 354,179 361,193 364,195 367,194 374,182 380,161 387,135 393,110 399,91 406,83 412,86 419,100 425,121 432,144 438,162 445,171 451,168 457,152 464,126 470,93 477,60 480,45" fill="none" stroke="#185FA5" stroke-width="2.5"/><circle cx="448" cy="165" r="6" fill="#C0507A"/><circle cx="364" cy="195" r="3.5" fill="#1D9E75"/><g font-family="sans-serif" font-size="11.5" text-anchor="middle" fill="#33312E"><text x="155" y="24">convex — one bowl</text><text x="385" y="24">non-convex — many dips</text></g><g font-family="sans-serif" font-size="10" fill="#9C9B95"><text x="155" y="216" text-anchor="middle">every dip is the global minimum</text><text x="448" y="150" text-anchor="middle" fill="#A32D2D">local min (stuck!)</text><text x="364" y="214" text-anchor="middle" fill="#1D9E75">global min</text></g></svg></div>

## 5. Batch, stochastic & mini-batch

- **Batch GD** — gradient over the *whole* dataset each step. Accurate but slow/memory-heavy.
- **Stochastic GD (SGD)** — one example at a time. Noisy but fast; the noise can even help
  escape shallow minima.
- **Mini-batch** — a small batch (32–512). The practical default — efficient on GPUs and a
  good noise/speed balance. (This is how every neural net trains.)

## 6. Momentum & adaptive methods

Vanilla GD struggles in **ravines** (steep one way, shallow another) and on flat regions.
Fixes:
- **Momentum** — accumulate a velocity so consistent directions build speed and
  oscillations cancel.
- **RMSProp / Adagrad** — per-parameter learning rates from recent gradient sizes.
- **Adam** — momentum + per-parameter rates; the **default optimizer for deep learning**.

## 7. Local minima, saddpoints & plateaus

In high dimensions, truly *bad* local minima are rare; **saddle points** (up in some
directions, down in others) and **flat plateaus** are the real obstacles — momentum and
adaptive methods are designed to push through them.

## 8. Beyond first order (brief)

**Newton's method** uses curvature (the Hessian) to take smarter steps — powerful but
$O(n^2)$–$O(n^3)$ in the number of parameters, so it's impractical for million-parameter
nets. First-order methods (SGD/Adam) dominate at scale.

## 9. Convexity, precisely

A function is **convex** if the chord between any two points on its graph lies *above*
the graph — equivalently (smooth case) its Hessian is positive semidefinite (curves up
everywhere). The payoff: **every local minimum is global**, so gradient descent can't get
stuck. MSE (linear regression), the logistic loss, and the SVM objective are convex;
neural-net losses are **not** (but GD still works well in practice).

## 10. Conditioning — why ravines are slow

The **condition number** (ratio of largest to smallest Hessian eigenvalue) measures how
*stretched* the bowl is. A high condition number = a long, narrow **ravine**: the step
size is capped by the **steep** direction while you crawl along the **shallow** one, so
GD **zig-zags** (the anisotropy slider in the playground). Cures: **feature scaling**
(M7), **momentum**, and adaptive/preconditioned methods.

## 11. Momentum & Adam — the equations

- **Momentum:** $v \leftarrow \beta v - \eta\nabla L,\quad \theta \leftarrow \theta + v$.
  Velocity builds along consistent directions and cancels oscillations across the ravine.
- **Adam:** track exponential moving averages of the gradient's 1st moment $m$ and 2nd
  moment $v$, bias-correct them, then
  $\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ — momentum **plus** a
  per-parameter learning rate. The default optimizer for deep nets.

## 12. Learning-rate schedules

A constant $\eta$ is rarely best. Common schedules: **warmup** (ramp up early to avoid
instability), then **decay** (step, exponential, or **cosine**) to settle into a minimum.
Large-scale LLM training is mostly warmup + cosine decay.

## 13. The convergence condition (worked)

For $L(w)=w^2$ the update is $w\leftarrow w-\eta\cdot 2w=(1-2\eta)\,w$. It converges iff
$|1-2\eta|<1$, i.e. $0<\eta<1$; at $\eta=0.5$ it hits the minimum in **one** step; for
$\eta>1$ it **diverges** — exactly what the playground shows. In general, GD on a smooth
loss converges for $\eta < 2/L$, where $L$ is the curvature (largest Hessian eigenvalue).

## 14. Constrained optimization & Lagrange multipliers

Real problems often **minimize subject to constraints** ("minimize risk *but keep the budget
within $B$*"). Two cases.

**Equality constraint.** To minimize $f(\mathbf x)$ subject to $g(\mathbf x)=0$, the optimum
sits where you can't lower $f$ without leaving the constraint — geometrically, where a level
set of $f$ is **tangent** to the constraint, so $\nabla f$ is **parallel** to $\nabla g$:
$\nabla f=\lambda\nabla g$ for a **Lagrange multiplier** $\lambda$. Equivalently, make the
**Lagrangian** $\mathcal L=f-\lambda g$ stationary in both $\mathbf x$ and $\lambda$.

*Worked:* minimize $x^2+y^2$ s.t. $x+y=1$. Then $\nabla f=(2x,2y)$ and $\nabla g=(1,1)$, so
$2x=2y=\lambda$ with $x+y=1$ give $x=y=\tfrac12$ — the closest point on the line to the origin:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 300 250" style="width:100%;max-width:300px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Minimizing x squared plus y squared subject to x plus y equals 1: concentric level circles centered at the origin are tangent to the constraint line at (0.5, 0.5), where the gradient of f is parallel to the gradient of g."><defs><marker id="la14" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#0E5E45"/></marker></defs><rect x="1" y="1" width="298" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><circle cx="70" cy="210" r="60" fill="none" stroke="#E2E2DA" stroke-width="1.3"/><circle cx="70" cy="210" r="92" fill="none" stroke="#1D9E75" stroke-width="1.8"/><circle cx="70" cy="210" r="120" fill="none" stroke="#E2E2DA" stroke-width="1.3"/><line x1="205" y1="210" x2="70" y2="75" stroke="#185FA5" stroke-width="2"/><circle cx="135" cy="145" r="4" fill="#0E5E45"/><line x1="135" y1="145" x2="156" y2="124" stroke="#0E5E45" stroke-width="2" marker-end="url(#la14)"/><circle cx="70" cy="210" r="3" fill="#33312E"/><g font-family="sans-serif" font-size="11"><text x="150" y="86" fill="#185FA5">x + y = 1</text><text x="160" y="118" fill="#0E5E45">∇f ∥ ∇g</text><text x="84" y="164" fill="#0E5E45">optimum</text></g></svg></div>

**Inequality constraints** ($g_i(\mathbf x)\le0$) add the **KKT conditions**, the workhorse of
constrained ML:

- **stationarity** $\nabla f+\sum_i\lambda_i\nabla g_i=\mathbf 0$,
- **primal feasibility** $g_i\le0$,
- **dual feasibility** $\lambda_i\ge0$,
- **complementary slackness** $\lambda_i\,g_i=0$ — each constraint is either *inactive*
  ($g_i<0,\ \lambda_i=0$) or *active* ($g_i=0,\ \lambda_i>0$).

This is exactly how the **SVM** (M4) is derived: the points whose margin constraint is
*active* are the **support vectors** ($\lambda_i>0$); every other point has $\lambda_i=0$ and
could be deleted without moving the boundary. The same duality links **penalties and
constraints**: adding $\lambda\lVert\mathbf w\rVert^2$ to a loss is the Lagrangian of "minimize
the loss s.t. $\lVert\mathbf w\rVert^2\le C$" — so a ridge/lasso $\lambda$ *is* a Lagrange
multiplier (X1 §11, M6).

## 15. Subgradients — optimizing non-smooth losses

Some losses have kinks (no derivative): the **hinge loss** (SVM) at the margin, **L1 /
lasso** at 0, **ReLU** at 0. A **subgradient** generalizes the gradient at such points
(any slope lying below the function), and **subgradient descent** still converges — it's
how the M4 SVM playground trains.

## 16. The optimizer family — a progression

Each one fixes a flaw of the previous:
- **SGD** — noisy mini-batch gradients.
- **+ Momentum** — accumulate velocity to power through ravines.
- **Nesterov** — "look-ahead" momentum (peek before stepping).
- **Adagrad** — per-parameter rates from accumulated squared grads (great for sparse
  features, but the rate decays to 0).
- **RMSProp** — Adagrad with a *moving* average (fixes the decay).
- **Adam** — RMSProp + momentum + bias correction; the default.
- **AdamW** — Adam with *decoupled* weight decay; today's LLM default.

## 17. Convergence rates (how fast?)

For a convex, smooth loss: plain GD converges as $O(1/t)$, **Nesterov-accelerated** as
$O(1/t^2)$, and for **strongly convex** losses GD is **linear** (error $\times\rho^t$).
SGD trades a noisier $O(1/\sqrt t)$ for cheap steps. **Conditioning** (§10) sets the
constant — ill-conditioned ⇒ slow.

## 18. Deep-net pathologies — vanishing/exploding gradients

In deep or recurrent nets the gradient is a **product of many Jacobians** (X2 §17), so it
can **vanish** ($\to0$: early layers stop learning) or **explode** ($\to\infty$: training
blows up). Fixes: good **initialization** (He/Xavier), **normalization** (batch/layer
norm), residual connections, ReLU-family activations, and **gradient clipping** (cap the
gradient norm). These are *why* deep nets are trainable at all.

## 19. Second-order & quasi-Newton (brief)

**Newton's method** uses the Hessian for curvature-aware steps (fast near the optimum)
but costs $O(n^3)$. **Quasi-Newton** methods (**BFGS, L-BFGS**) approximate the Hessian
cheaply and are excellent for **small/medium** smooth problems (e.g. fitting logistic
regression). At neural-net scale, first-order methods (Adam/SGD) still win.

## 20. Connection to ML & ANN

This is **how every model in the lab learns**: compute the gradient of the loss (by
calculus/backprop, X2 / ANN), then apply the update in §2. Logistic regression, SVM, and
neural-net training are all gradient descent on different losses. The learning rate,
momentum, and Adam you meet here are exactly the dials in ANN Tier 1+.
"""

_X4_TASKS = r"""
### Warm-up — in the Playground tab
1. Increase the **learning rate** until the path **diverges** — what's the largest stable
   rate? Now make it tiny and watch it crawl.
2. Raise the **anisotropy** (make the bowl a ravine) — see GD **zig-zag**. Why does
   feature scaling (M7) help here?

### Pencil & paper
3. Write the GD update for $L(w)=w^2$; with $\eta=0.1$ and $w_0=4$, compute $w_1,w_2,w_3$.
4. For what learning rates does GD on $L(w)=w^2$ converge? (Hint: the update is
   $w\leftarrow w(1-2\eta)$.)

### Code
5. Implement gradient descent on $f(x,y)=x^2+10y^2$ from a fixed start; plot the path on a
   contour and the loss vs. step.
6. Add **momentum** and compare the number of steps to converge.

### Bridge
7. Connect to ANN Tier 1: backprop computes ∇L, this update changes the weights. SGD vs
   batch — which does a neural net use and why?
"""

_X4_REFS = r"""
### Video & books
- **3Blue1Brown** — [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent) (in the neural-net series).
- Deisenroth, Faisal & Ong — *Mathematics for ML*, ch. 7 (optimization) — [free PDF](https://mml-book.github.io/).
- Ruder (2016) — *An overview of gradient descent optimization algorithms* (SGD → Adam).

### In this lab
- ANN: this is how weights are learned (Tier 1, backprop + the update). Math: X2 (the gradient).
- ML: M1 §4 (GD for regression), M2 §6 (GD for logistic regression).
"""

_X4_QUIZ = [
    Question("The gradient descent update is…",
             ["θ ← θ + η∇L", "θ ← θ − η∇L", "θ ← −∇L", "θ ← θ·∇L"], 1,
             "Step opposite the gradient (downhill), scaled by the learning rate η."),
    Question("If the learning rate is too large, gradient descent tends to…",
             ["converge faster with no downside", "overshoot and diverge / oscillate",
              "always find the global minimum", "ignore the gradient"], 1,
             "Too-large steps overshoot the minimum and the loss can blow up."),
    Question("A convex loss guarantees that…",
             ["there are many minima", "every local minimum is the global minimum",
              "gradient descent fails", "the gradient is always zero"], 1,
             "Convex = single bowl, so GD reaches the global optimum (linear/logistic/SVM)."),
    Question("Mini-batch SGD is the practical default because it…",
             ["uses the whole dataset each step", "balances speed (GPU-friendly batches) and gradient-noise",
              "needs no learning rate", "only works for convex losses"], 1,
             "Mini-batches are efficient on GPUs and trade accuracy of the gradient for speed."),
    Question("Adam is popular because it…",
             ["is second-order", "combines momentum with per-parameter adaptive learning rates",
              "needs no gradient", "only works on images"], 1,
             "Adam = momentum + adaptive per-parameter rates; the deep-learning default."),
    Question("In high-dimensional neural-net training, the bigger obstacle is usually…",
             ["bad local minima everywhere", "saddle points and flat plateaus",
              "the convexity", "too few parameters"], 1,
             "Saddles/plateaus dominate; momentum/adaptive methods help escape them."),
    Question("Newton's method isn't used for large neural nets because…",
             ["it ignores the gradient", "computing/inverting the Hessian is too expensive at scale",
              "it only works in 1-D", "it has no learning rate"], 1,
             "Second-order methods cost O(n²)–O(n³) in parameters — infeasible for millions of weights."),
]

OPTIMIZATION = Lesson("optimization", "Optimization", _X4_THEORY, _X4_QUIZ, _X4_TASKS, _X4_REFS)


# ===========================================================================
# X5 — Information theory
# ===========================================================================

_X5_THEORY = r"""
## 1. Why information theory for ML

It quantifies **uncertainty** and **surprise** — and it's the source of the
**classification losses** (cross-entropy) and the splitting criterion of decision trees
(information gain). A little of it explains a lot of ML.

## 2. Information / surprise

The information in an event of probability $p$ is $-\log p$. Rare events are
**surprising** (carry many bits); certain events ($p=1$) carry **zero** information. Use
$\log_2$ for **bits**, $\ln$ for **nats**.

## 3. Entropy — expected surprise

The **entropy** of a distribution is the average surprise:
$$ H(p) = -\sum_x p(x)\log p(x). $$
It's **maximal for a uniform** distribution (maximum uncertainty) and **0** when one
outcome is certain. A fair coin has $H=1$ bit; a two-headed coin has $H=0$. Entropy =
the irreducible number of bits needed, on average, to encode outcomes. For a coin with
bias $p$, entropy peaks at $p=0.5$ (most uncertain) and falls to 0 at either certain end:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 420 270" style="width:100%;max-width:420px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Binary entropy as a function of the coin bias p: a dome peaking at 1 bit when p is 0.5 (maximum uncertainty) and dropping to 0 at p=0 and p=1 (a certain outcome)."><rect x="1" y="1" width="418" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="50" y1="225" x2="408" y2="225" stroke="#C9C8C1" stroke-width="1.2"/><line x1="50" y1="40" x2="50" y2="225" stroke="#C9C8C1" stroke-width="1.2"/><line x1="225" y1="54" x2="225" y2="225" stroke="#D9D8D1" stroke-width="1" stroke-dasharray="4 3"/><polyline points="50,223 56,203 62,188 68,175 74,163 80,153 86,143 92,135 98,127 104,119 110,112 115,106 121,100 127,94 133,89 139,85 145,80 151,76 157,73 163,70 169,67 175,64 181,62 187,60 192,58 198,56 204,55 210,54 216,54 222,54 228,54 234,54 240,54 246,55 252,56 258,58 263,60 269,62 275,64 281,67 287,70 293,73 299,76 305,80 311,85 317,89 323,94 329,100 335,106 340,112 346,119 352,127 358,135 364,143 370,153 376,163 382,175 388,188 394,203 400,223" fill="none" stroke="#185FA5" stroke-width="2.6"/><g font-family="sans-serif" font-size="11" fill="#6B6A66"><text x="46" y="58" text-anchor="end">1 bit</text><text x="46" y="229" text-anchor="end">0</text><text x="50" y="240" text-anchor="middle">0</text><text x="225" y="240" text-anchor="middle">0.5</text><text x="400" y="240" text-anchor="middle">1</text><text x="225" y="258" text-anchor="middle">coin bias p</text><text x="225" y="50" text-anchor="middle" fill="#185FA5">max uncertainty</text></g></svg></div>

## 4. Cross-entropy — the classification loss

**Cross-entropy** measures the cost of using a *predicted* distribution $q$ when the
*true* one is $p$:
$$ H(p,q) = -\sum_x p(x)\log q(x). $$
It is minimized (equal to $H(p)$) exactly when $q=p$. With one-hot truth $p$, it reduces
to $-\log q(\text{correct class})$ — the **cross-entropy / log loss** of M2 and ANN. So
training a classifier = making its predicted distribution match the true labels. Sweeping
the prediction $q$ (true bias $p=0.7$), the loss bottoms out **exactly at $q=p$**:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 420 270" style="width:100%;max-width:420px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Cross-entropy as a function of the predicted probability q when the true bias is 0.7. The curve is a valley reaching its minimum exactly at q equals 0.7, and rising steeply as q approaches 0 (confident and wrong)."><rect x="1" y="1" width="418" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="50" y1="225" x2="408" y2="225" stroke="#C9C8C1" stroke-width="1.2"/><line x1="50" y1="40" x2="50" y2="225" stroke="#C9C8C1" stroke-width="1.2"/><line x1="298" y1="50" x2="298" y2="225" stroke="#1D9E75" stroke-width="1.3" stroke-dasharray="5 3"/><polyline points="50,45 56,45 62,56 68,71 74,83 80,92 86,100 92,107 97,113 103,119 109,124 115,128 121,132 127,135 133,139 139,142 145,145 151,147 157,150 163,152 169,154 175,156 181,158 186,160 192,162 198,163 204,165 210,166 216,167 222,168 228,169 234,170 240,171 246,172 252,173 258,173 264,174 269,174 275,175 281,175 287,175 293,175 299,175 305,175 311,175 317,175 323,174 329,174 335,173 341,172 347,171 353,170 358,168 364,166 370,163 376,160 382,156 388,150 394,142 400,129" fill="none" stroke="#185FA5" stroke-width="2.6"/><circle cx="298" cy="175" r="4" fill="#1D9E75"/><g font-family="sans-serif" font-size="11" fill="#6B6A66"><text x="298" y="46" text-anchor="middle" fill="#1D9E75">min at q = p</text><text x="225" y="256" text-anchor="middle">predicted probability q  (true p = 0.7)</text><text x="44" y="52" text-anchor="end">loss</text><text x="92" y="86" fill="#A32D2D">confident &amp; wrong</text></g></svg></div>

## 5. KL divergence — distance between distributions

The extra cost of using $q$ instead of $p$:
$$ D_{\mathrm{KL}}(p\Vert q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p) \ge 0. $$
It's 0 iff $p=q$, and **asymmetric** ($D(p\Vert q)\ne D(q\Vert p)$), so it's a
"divergence", not a true distance. Since $H(p)$ is fixed by the data, **minimizing
cross-entropy = minimizing KL to the true distribution = maximizing likelihood** (X3) —
three views of the same training objective.

## 6. Units & the coding interpretation

Entropy in **bits** ($\log_2$) is the average number of yes/no questions — or compressed
bits — needed to pin down an outcome. Shannon's **source-coding theorem** says you can't
compress below $H$ bits on average. (**Nats** use $\ln$.) A fair 8-sided die has $H=3$
bits; you genuinely need 3 bits to encode each roll.

## 7. Joint, conditional entropy & mutual information

- **Conditional entropy** $H(Y\mid X)$ = the uncertainty left in $Y$ once you know $X$.
- **Mutual information** $I(X;Y)=H(Y)-H(Y\mid X)\ge 0$ = how many bits knowing $X$ saves
  about $Y$; it's $0$ iff they're independent. MI drives **feature selection**, and it's
  the principle behind decision-tree **information gain** (M3).

## 8. Cross-entropy = negative log-likelihood

For one-hot truth, cross-entropy reduces to $-\log q(\text{true class})$ — the
**negative log-likelihood** of the data under the model. So three things you've met are
*the same objective*: **minimize cross-entropy** (M2/ANN) = **maximize likelihood**
(X3 MLE) = **minimize KL to the truth** (§5).

## 9. Forward vs. reverse KL

KL is asymmetric and the direction matters. **Forward** $D_{\mathrm{KL}}(p\Vert q)$ (used
in MLE / supervised learning) is **mode-covering** — $q$ spreads to cover all of $p$'s
mass. **Reverse** $D_{\mathrm{KL}}(q\Vert p)$ (used in variational inference and some RL)
is **mode-seeking** — $q$ locks onto a single mode. This subtlety shapes a lot of
generative-model behavior.

## 10. Perplexity — how LLMs are scored

Language models report **perplexity** $= 2^{H(p,q)}$ (or $e^{\text{cross-entropy}}$):
the *effective number of equally-likely next tokens* the model is choosing among. Lower
is better. Minimizing an LLM's cross-entropy loss **is** minimizing its perplexity — the
direct line from this lesson to how GPT-class models are trained and evaluated.

## 11. A worked example

True label one-hot $p=[1,0]$, model $q=[0.7,0.3]$ ⇒ cross-entropy $=-\log_2 0.7\approx \mathbf{0.51}$ bits. Sure-and-right ($q=[1,0]$) ⇒ **0**; sure-and-wrong ($q=[0.01,0.99]$)
⇒ $-\log_2 0.01\approx\mathbf{6.6}$ bits. That steep penalty for confident errors is what
trains the classifier.

## 12. Differential entropy & the max-entropy Gaussian

For *continuous* variables, **differential entropy** $h(p)=-\int p(x)\log p(x)\,dx$. A
key fact: among all distributions with a given mean and variance, the **Gaussian has
maximum entropy** — the "least-assuming" choice, another reason it's the default noise
model (X3).

## 13. Why cross-entropy ≥ entropy (Gibbs' inequality)

$H(p,q)\ge H(p)$, with equality iff $q=p$ — **Gibbs' inequality**, equivalently
$D_{\mathrm{KL}}(p\Vert q)\ge0$, which follows from **Jensen's inequality** (log is
concave). It is *why* minimizing cross-entropy/KL drives $q\to p$, and why the X5
playground's curve bottoms out exactly at $q=p$.

## 14. Chain rule of entropy & mutual information as a KL

$H(X,Y)=H(X)+H(Y\mid X)$ (uncertainty adds up sequentially). And mutual information is
itself a KL divergence: $I(X;Y)=D_{\mathrm{KL}}\big(p(x,y)\,\Vert\,p(x)p(y)\big)$ — the
distance of the joint from independence — which makes it symmetric and $\ge 0$.

## 15. The softmax + cross-entropy gradient (derived)

For logits $\mathbf z$, $\mathbf p=\mathrm{softmax}(\mathbf z)$, one-hot target $\mathbf y$, loss $L=-\sum_k y_k\log p_k$:
$$ \frac{\partial L}{\partial z_i}=p_i-y_i. $$
The messy softmax Jacobian and the $\log$ cancel into a clean "**prediction − target**" —
the reason softmax and cross-entropy are *always* paired (M2, ANN e08). Worth deriving once.

## 16. Information gain in trees (worked)

Decision trees (M3) split to **maximize information gain** = entropy reduction. A node
with classes $[8,2]$ has $H=-0.8\log_2 0.8-0.2\log_2 0.2\approx 0.72$ bits. Split into
pure $[6,0]$ ($H=0$) and $[2,2]$ ($H=1$): weighted child entropy
$\tfrac{6}{10}(0)+\tfrac{4}{10}(1)=0.4$, so **gain $=0.72-0.4=0.32$ bits**. (Compare the
Gini version in M3 §3.)

## 17. Label smoothing & f-divergences (brief)

**Label smoothing** softens one-hot targets (e.g. 0.9 / 0.1 instead of 1 / 0) — an
information-theoretic regularizer that improves calibration and generalization. And KL is
one member of the broader **f-divergence** family (total variation, Jensen–Shannon — the
symmetric cousin used in GANs).

## 18. Connections across the lab

- **Cross-entropy loss** + **softmax** — M2 multiclass, ANN output layers.
- **Information gain** — decision-tree splits (M3) reduce entropy.
- **KL divergence** — variational methods, VAEs, and the regularizer in RLHF/PPO for LLMs.
- **Mutual information** — how much knowing one variable reduces uncertainty about
  another (feature selection, representation learning).
"""

_X5_TASKS = r"""
### Pencil & paper
1. Compute the entropy (bits) of a fair coin, a 4-sided fair die, and a coin with p=0.9.
2. With true label one-hot p = [1,0] and prediction q = [0.7, 0.3], compute the cross-entropy.
3. Show that cross-entropy H(p,q) = H(p) + D_KL(p‖q), so minimizing it minimizes the KL term.

### Code
4. Implement entropy, cross-entropy, and KL in NumPy; verify D_KL(p‖q) ≥ 0 and = 0 when p=q.
5. Plot cross-entropy loss −log q for a true positive as q goes 0→1 (see it explode near 0).

### Bridge
6. Connect: cross-entropy here ↔ M2 logistic/softmax loss ↔ ANN e08 ↔ MLE (X3 §7).
7. Decision trees (M3) split to maximize information gain = entropy reduction — relate.
"""

_X5_REFS = r"""
### Books & video
- Deisenroth, Faisal & Ong — *Mathematics for ML* (info-theory appendix) — [free PDF](https://mml-book.github.io/).
- MacKay — *Information Theory, Inference, and Learning Algorithms* (free, the classic).
- **StatQuest** — entropy, cross-entropy & KL explained simply.

### In this lab
- ML: M2 (cross-entropy), M3 (information gain). ANN: e08 (cross-entropy gradient), softmax.
- Math: X3 (MLE = minimizing KL).
"""

_X5_QUIZ = [
    Question("The information (surprise) of an event with probability p is…",
             ["p", "−log p (rare events carry more)", "1 − p", "p²"], 1,
             "Surprise = −log p; certain events (p=1) carry zero information."),
    Question("Entropy H(p) = −Σ p log p is largest when the distribution is…",
             ["a point mass (one certain outcome)", "uniform (maximum uncertainty)",
              "Gaussian", "negative"], 1,
             "Uniform = most uncertain = max entropy; a certain outcome has H=0."),
    Question("Cross-entropy H(p,q) is minimized when…",
             ["q is uniform", "q equals the true distribution p", "q is one-hot", "p is uniform"], 1,
             "H(p,q) ≥ H(p) with equality iff q = p — the goal of training a classifier."),
    Question("KL divergence D(p‖q) is…",
             ["always symmetric", "≥ 0, and 0 iff p = q (but not symmetric)",
              "a true distance metric", "negative for good models"], 1,
             "KL ≥ 0, zero only when the distributions match, and D(p‖q) ≠ D(q‖p)."),
    Question("Minimizing cross-entropy is equivalent to…",
             ["maximizing entropy", "minimizing KL to the true distribution / maximizing likelihood",
              "minimizing variance", "maximizing the margin"], 1,
             "Since H(p) is fixed, minimizing H(p,q) minimizes D_KL(p‖q) = maximizing likelihood."),
    Question("Decision trees choose splits to…",
             ["increase entropy", "maximize information gain (reduce entropy/impurity)",
              "maximize KL", "keep entropy constant"], 1,
             "Information gain = parent entropy − weighted child entropy; trees maximize it (M3)."),
]

INFORMATION = Lesson("information", "Information theory", _X5_THEORY, _X5_QUIZ, _X5_TASKS, _X5_REFS)


# ===========================================================================
# X6 — Numerical computing
# ===========================================================================

_X6_THEORY = r"""
## 1. Why numerical computing

Math on a computer is **not exact**, and the gap is where subtle ML bugs live: NaNs in
training, a loss that explodes, results that differ across runs. A little numerical
literacy saves hours.

## 2. Floating point

Computers store reals in finite precision (IEEE-754): **float32** (~7 decimal digits) and
**float64** (~16). Most reals can't be represented exactly, so every operation **rounds**.
Famous consequence: `0.1 + 0.2 == 0.30000000000000004`. The smallest gap near 1.0 is
**machine epsilon** (~1e-7 for float32). Deep learning often uses float32 (or float16 /
bfloat16) for speed and memory — trading precision for throughput.

## 3. Overflow & underflow

$e^{1000}$ is `inf`; $e^{-1000}$ underflows to 0. Naive formulas explode:
$\sigma(z)=\frac{1}{1+e^{-z}}$ overflows for very negative $z$, and **softmax** overflows
for large logits. The fix is the **log-sum-exp trick**: subtract the max before
exponentiating, $\text{softmax}(z)_i=\frac{e^{z_i-\max z}}{\sum_j e^{z_j-\max z}}$. The
lab's `core/activations.py` already uses a sign-split stable sigmoid — worth reading.

## 4. Cancellation & stability

**Catastrophic cancellation**: subtracting two nearly-equal numbers throws away
significant digits (e.g. the naive variance formula $\mathbb E[x^2]-\mathbb E[x]^2$). Use
stable alternatives: `log1p(x)` for $\log(1+x)$, `expm1`, Welford's algorithm for
variance, and **log-space** for products of many small probabilities (sum logs instead of
multiplying — avoids underflow; this is why we use log-likelihood, X3).

## 5. Vectorization — the performance idea

Python loops are slow; **NumPy** pushes array math down to optimized C/BLAS that uses SIMD
and multiple cores. Express computation as **array operations / matrix multiplies**, not
loops — often 10–100× faster, and the same code maps onto **GPUs** (where a neural net's
matmuls run in parallel). Vectorization is the single most important practical-speed habit.

## 6. Gradient checking & determinism

- **Gradient checking** — verify analytic gradients against the **finite-difference**
  approximation $\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}$ (ANN e06). Pick
  $\varepsilon$ around 1e-5: too small drowns in float noise, too large adds truncation error.
- **Determinism** — fix random **seeds** (and beware that some GPU ops are non-deterministic)
  so runs are reproducible.

## 7. Float formats & mixed precision

| format | bits | approx. range | typical use |
|---|---|---|---|
| float64 | 64 | $10^{\pm308}$ | scientific computing, CPU default |
| float32 | 32 | $10^{\pm38}$ | the standard ML default |
| float16 | 16 | $\sim 6\times10^{4}$ | fast/small, but **overflows easily** |
| bfloat16 | 16 | $10^{\pm38}$ | fp32's range with less precision — the deep-learning favorite |

**Mixed-precision** training does the math in fp16/bf16 (fast, memory-light on GPUs/TPUs)
while keeping an fp32 master copy of the weights — standard in modern deep learning.

## 8. The log-sum-exp trick, derived

To compute $\log\sum_i e^{z_i}$ without overflow, pull out the max $m=\max_i z_i$:
$$ \log\sum_i e^{z_i} = m + \log\sum_i e^{z_i - m}. $$
Now every exponent $z_i-m\le 0$ (no overflow), and at least one term is $e^0=1$ (the sum
can't underflow to 0). Stable **softmax** and **cross-entropy** are built on exactly this
identity — including `core/activations.py`:

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 520 195" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Naive softmax on large logits 1000, 1001, 1002 overflows: exp gives infinity and infinity over infinity is NaN. Subtracting the max first gives logits -2, -1, 0, exp gives finite values, and softmax returns a valid probability distribution."><rect x="1" y="1" width="518" height="193" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="18" y="30" width="232" height="150" rx="8" fill="#FFFFFF" stroke="#E2E2DA"/><rect x="270" y="30" width="232" height="150" rx="8" fill="#FFFFFF" stroke="#E2E2DA"/><text x="134" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#A32D2D">naive — overflow</text><text x="386" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D9E75">subtract the max (m = 1002)</text><g font-family="monospace" font-size="12"><text x="32" y="88" fill="#33312E">z = [1000,1001,1002]</text><text x="32" y="118" fill="#A32D2D">exp(z) = [inf,inf,inf]</text><text x="32" y="158" fill="#A32D2D">inf / inf = NaN  ✗</text><text x="284" y="88" fill="#33312E">z−m = [−2, −1, 0]</text><text x="284" y="118" fill="#33312E">exp = [.14, .37, 1.0]</text><text x="284" y="158" fill="#1D9E75">= [.09, .24, .67]  ✓</text></g></svg></div>

## 9. Summation & accumulation error

Float addition isn't associative, so adding many numbers accumulates rounding error and
the **order matters**. Summing a huge array naively loses precision; **Kahan summation**
(or pairwise summation, which NumPy uses) tracks the lost low-order bits. It matters
whenever you reduce over millions of values — losses, norms, gradients.

## 10. Broadcasting & memory layout

NumPy **broadcasting** combines arrays of compatible shapes without explicit loops or
copies — e.g. `X - X.mean(0)` standardizes each column in one vectorized line.
Understanding broadcasting (and row-major contiguous memory) is how you write fast,
correct array code and avoid silent shape bugs.

## 11. Randomness & reproducibility

Pseudo-random generators are **deterministic given a seed** — fix it
(`np.random.default_rng(seed)`) to reproduce splits, weight initializations, and results.
Caveats: global vs. local RNG state, and some **GPU kernels are non-deterministic** by
default, so bit-exact reproducibility on GPU needs extra flags.

## 12. IEEE-754 anatomy

A float stores three parts: **sign** (1 bit), **exponent**, and **mantissa** (fraction).
float32 = 1 + 8 + 23 bits, with value $(-1)^s\times 1.\text{mantissa}\times 2^{\text{exp}-127}$ — "scientific notation in binary": ~7 significant decimal digits,
range ~$10^{\pm38}$. Special bit-patterns encode $\pm\infty$, **NaN**, and subnormals.
The layout is *why* precision is **relative** (next section).

<div style="text-align:center;margin:0.6rem 0"><svg viewBox="0 0 700 165" style="width:100%;max-width:700px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The 32 bits of a float32: 1 sign bit, 8 exponent bits, and 23 mantissa (fraction) bits. Its value is minus-one to the sign, times 1 point mantissa, times two to the (exponent minus 127)."><rect x="1" y="1" width="698" height="163" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="20" y="28" font-family="sans-serif" font-size="12" fill="#33312E">float32 — 32 bits</text><rect x="20" y="40" width="20" height="36" fill="#FBEAF0" stroke="#C0507A"/><rect x="40" y="40" width="160" height="36" fill="#FBEAD6" stroke="#9A6A2A"/><rect x="200" y="40" width="440" height="36" fill="#E6F1FB" stroke="#5B8FC2"/><g stroke="#FFFFFF" stroke-width="1"><line x1="60" y1="41" x2="60" y2="75"/><line x1="80" y1="41" x2="80" y2="75"/><line x1="100" y1="41" x2="100" y2="75"/><line x1="120" y1="41" x2="120" y2="75"/><line x1="140" y1="41" x2="140" y2="75"/><line x1="160" y1="41" x2="160" y2="75"/><line x1="180" y1="41" x2="180" y2="75"/><line x1="240" y1="41" x2="240" y2="75"/><line x1="280" y1="41" x2="280" y2="75"/><line x1="320" y1="41" x2="320" y2="75"/><line x1="360" y1="41" x2="360" y2="75"/><line x1="400" y1="41" x2="400" y2="75"/><line x1="440" y1="41" x2="440" y2="75"/><line x1="480" y1="41" x2="480" y2="75"/><line x1="520" y1="41" x2="520" y2="75"/><line x1="560" y1="41" x2="560" y2="75"/><line x1="600" y1="41" x2="600" y2="75"/></g><text x="30" y="63" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#A32D2D">S</text><text x="120" y="63" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">exponent</text><text x="420" y="63" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#0C447C">mantissa (fraction)</text><g font-family="sans-serif" font-size="10" fill="#6B6A66" text-anchor="middle"><text x="30" y="94">1 bit</text><text x="120" y="94">8 bits</text><text x="420" y="94">23 bits</text></g><text x="350" y="128" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#33312E">value = (−1)ˢ × 1.mantissa × 2^(exponent − 127)</text><text x="350" y="148" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">≈ 7 significant decimal digits · range ≈ 10^±38</text></svg></div>

## 13. Machine epsilon, ulp & relative error

**Machine epsilon** $\varepsilon$ is the gap between 1.0 and the next float
(~$1.2\times10^{-7}$ for float32). Precision is **relative**: near a large number the
spacing (one **ulp** — unit in the last place) is large, so `1e8 + 1.0 == 1e8` in
float32 (the 1.0 is lost). Always reason in *relative* error ($\sim\varepsilon$), not
absolute.

## 14. Conditioning vs. stability

Two distinct ideas: a **problem** is *ill-conditioned* if tiny input changes cause large
output changes (large **condition number**); an **algorithm** is *unstable* if it
amplifies rounding error. You want stable algorithms — and no algorithm rescues an
ill-conditioned problem. Inverting a near-singular matrix is the classic ill-conditioned
case (next).

## 15. Solving linear systems — never invert

To solve $A\mathbf x=\mathbf b$ (e.g. the normal equations $X^\top X\mathbf w=X^\top\mathbf y$ from M1), **don't compute $A^{-1}$** — it's slower and less stable. Use a
**factorization**: **Cholesky** (symmetric positive-definite, like $X^\top X$), **LU**,
or — best for least squares — **QR / SVD** on $X$ directly (avoids forming the
ill-conditioned $X^\top X$). `np.linalg.lstsq` / `solve` do this; calling `inv` is a code
smell.

## 16. Quantization & low-precision inference

Beyond fp16/bf16 *training*, deployed models often use **int8** (or 4-bit) **quantization**:
map weights/activations to small integers with a scale factor. It shrinks memory ~4× and
speeds inference at a small accuracy cost — how LLMs run on phones and laptops. Modern
numerics is increasingly about using *less* precision, cleverly.

## 17. Determinism & parallel reduction

Floating-point addition is **not associative** ($(a+b)+c\ne a+(b+c)$ in general), so a
**parallel** sum on a GPU that reorders additions can give slightly different results
run-to-run. Combined with non-deterministic kernel scheduling, this is why bit-exact GPU
reproducibility needs explicit deterministic flags — a fixed seed alone isn't enough.

## 18. Connection to ML & ANN

Stable losses (log-sum-exp softmax/cross-entropy), float32/mixed-precision training,
vectorized/GPU matmuls, gradient checking, and seeding are all everyday deep-learning
concerns. Frameworks (NumPy, PyTorch) exist largely to make this fast **and** numerically
safe — but you still need to know where the cliffs are.
"""

_X6_TASKS = r"""
### Try it (Sandbox or here)
1. Print `0.1 + 0.2` and `0.1 + 0.2 == 0.3`. Explain.
2. Compute `np.exp(1000)` (inf) and a naive softmax of `[1000, 1001, 1002]` (NaNs); then
   implement the **log-sum-exp** stable version and confirm it works.
3. Compare a Python loop vs. a NumPy vectorized dot product on a large array (time both).

### Pencil & paper
4. Why does subtracting two nearly-equal floats lose precision? Give the variance-formula example.
5. Why do we sum log-probabilities instead of multiplying probabilities?

### Bridge
6. Read `core/activations.py`'s stable sigmoid; relate to §3. Connect §6 to ANN e06
   (gradient checking) and to reproducibility (seeds) used throughout the lab.
"""

_X6_REFS = r"""
### Classics & docs
- Goldberg (1991) — *What Every Computer Scientist Should Know About Floating-Point Arithmetic*.
- NumPy docs — [vectorization / broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
- The "log-sum-exp trick" (many good write-ups) — stable softmax/cross-entropy.

### In this lab
- ANN: `core/activations.py` (stable sigmoid), e06 (gradient checking). Math: X3 (log-likelihood).
"""

_X6_QUIZ = [
    Question("Why is 0.1 + 0.2 not exactly 0.3 on a computer?",
             ["Python is buggy", "floating-point uses finite binary precision, so most decimals round",
              "0.3 is undefined", "integers overflow"], 1,
             "IEEE-754 can't represent 0.1/0.2/0.3 exactly; each op rounds."),
    Question("The log-sum-exp trick (subtract the max before exponentiating) prevents…",
             ["slow training", "overflow/underflow in softmax / sigmoid",
              "overfitting", "data leakage"], 1,
             "Subtracting max keeps exponents in a safe range — numerically stable softmax."),
    Question("Vectorization (NumPy array ops instead of Python loops) helps because…",
             ["it uses less memory always", "it runs optimized C/BLAS (SIMD, multicore) and maps to GPUs",
              "it avoids floating point", "it removes the need for a model"], 1,
             "Array operations push work to fast compiled kernels — often 10–100× faster, GPU-friendly."),
    Question("Summing log-probabilities instead of multiplying probabilities avoids…",
             ["overfitting", "numerical underflow from multiplying many small numbers",
              "the need for data", "the chain rule"], 1,
             "Products of tiny probabilities underflow to 0; logs turn them into a safe sum."),
    Question("In gradient checking, choosing ε around 1e-5 balances…",
             ["bias vs. variance", "floating-point noise (too small) vs. truncation error (too large)",
              "precision vs. recall", "speed vs. memory"], 1,
             "Too-small ε is swamped by float noise; too-large ε adds finite-difference truncation error."),
    Question("Deep learning often trains in float32 / float16 because…",
             ["it's more accurate than float64", "it trades some precision for speed and memory",
              "it avoids rounding", "it's required by calculus"], 1,
             "Lower precision is faster and lighter on GPUs; mixed precision is standard in modern training."),
]

NUMERICAL = Lesson("numerical", "Numerical computing", _X6_THEORY, _X6_QUIZ, _X6_TASKS, _X6_REFS)
