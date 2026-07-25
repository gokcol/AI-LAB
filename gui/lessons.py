"""Lesson content + a tiny self-check quiz engine for the GUI.

A Lesson bundles deep theory, a quiz, hands-on tasks, and references for one
concept. The Playground renders these alongside the interactive bench. Add more
lessons (perceptron, activations, ...) by appending to REGISTRY — the structure
is reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass
class Question:
    prompt: str
    options: list[str]
    answer: int          # index into options
    explanation: str


@dataclass
class Lesson:
    key: str
    title: str
    theory: str          # markdown (LaTeX with $...$ / $$...$$)
    quiz: list[Question]
    tasks: str           # markdown
    references: str       # markdown
    intro_blocks: list = field(default_factory=list)  # [{"md":..} | {"svg":..,"caption":..}]


def render_quiz(questions: list[Question], prefix: str) -> None:
    """Immediate-feedback quiz. Picking an option reveals correct/explanation."""
    top = st.container()
    if st.button("Reset answers", key=f"{prefix}_reset"):
        for i in range(len(questions)):
            st.session_state.pop(f"{prefix}_q{i}", None)
        st.rerun()

    correct = answered = 0
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i + 1}. {q.prompt}**")
        choice = st.radio(
            "Your answer:", q.options, index=None,
            key=f"{prefix}_q{i}", label_visibility="collapsed",
        )
        if choice is not None:
            answered += 1
            if q.options.index(choice) == q.answer:
                correct += 1
                st.success(f"Correct — {q.explanation}")
            else:
                right = q.options[q.answer]
                st.error(f"Not quite. Answer: **{right}**. {q.explanation}")
        st.divider()

    with top:
        if answered:
            st.metric("Score", f"{correct}/{len(questions)}",
                      help=f"{answered} of {len(questions)} answered")
        else:
            st.caption("Pick an answer to get instant feedback. Nothing is graded — this is for you.")


def render_intro(blocks: list) -> None:
    """Render Part 0 intro blocks: markdown text interleaved with inline SVG figures."""
    for block in blocks:
        if "md" in block:
            st.markdown(block["md"])
        elif "svg" in block:
            st.markdown(
                f"<div style='text-align:center;margin:0.4rem 0'>{block['svg']}</div>",
                unsafe_allow_html=True,
            )
            if block.get("caption"):
                st.caption(block["caption"])


def predict(prompt: str, reveal: str) -> None:
    """Predict-then-reveal: pose a question and hide the answer behind a click, so the
    learner commits to a guess *before* seeing the result. The wrong guesses are what
    make the idea stick (the generation effect). Reusable across every page."""
    st.markdown(f"🔮 **Predict first —** {prompt}")
    with st.expander("Reveal the answer"):
        st.markdown(reveal)


def solution(body: str, *, label: str = "Show worked solution") -> None:
    """A collapsed worked solution — lets a solo learner attempt a task, then self-check.
    Without this, the exercises are open-loop (no instructor to verify against)."""
    with st.expander(f"✅ {label}"):
        st.markdown(body)


def render_lesson_content(lesson, solutions=None) -> None:
    """Render the 4 standard content tabs for a lesson (no interactive playground).
    Pass `solutions` as a list of (label, body) tuples to add collapsible worked
    answers under the Tasks tab."""
    t1, t2, t3, t4 = st.tabs(["📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"])
    with t1:
        st.markdown(lesson.theory, unsafe_allow_html=True)
    with t2:
        st.subheader("Self-check")
        st.caption("Instant feedback, no grading.")
        render_quiz(lesson.quiz, prefix=lesson.key)
    with t3:
        st.subheader("Tasks")
        st.markdown(lesson.tasks)
        if solutions:
            st.divider()
            st.markdown("#### ✅ Worked solutions")
            st.caption("Attempt each first, then check.")
            for label, body in solutions:
                solution(body, label=label)
    with t4:
        st.subheader("Reading & references")
        st.markdown(lesson.references)


# ===========================================================================
# Lesson: the single artificial neuron
# ===========================================================================

_THEORY = r"""
## 1. What a neuron computes

A neuron maps an input vector $\mathbf{x}=(x_1,\dots,x_n)$ to a single scalar in
**two stages**:

$$ z = \mathbf{w}\cdot\mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b, \qquad a = \varphi(z) $$

- $z$ — the **pre-activation** ("logit", "net input"): an affine function of the inputs.
- $\varphi$ — the **activation** (the nonlinearity).
- $a$ — the neuron's **output**.

In one picture: each input is scaled by its weight, the results are **summed** with the
bias to make $z$, and $\varphi$ bends that single number into the output $a$.

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 600 280" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Schematic of one neuron: each input times its weight is summed with a bias to give z, then an activation phi gives the output a."><defs><marker id="nnah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#33312E"/></marker></defs><rect x="1" y="1" width="598" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g stroke="#33312E" stroke-width="1.6" fill="none"><line x1="116" y1="68" x2="262" y2="126" marker-end="url(#nnah)"/><line x1="116" y1="140" x2="258" y2="140" marker-end="url(#nnah)"/><line x1="116" y1="212" x2="262" y2="154" marker-end="url(#nnah)"/><line x1="300" y1="232" x2="300" y2="182" marker-end="url(#nnah)"/><line x1="340" y1="140" x2="394" y2="140" marker-end="url(#nnah)"/><line x1="464" y1="140" x2="524" y2="140" marker-end="url(#nnah)"/></g><g font-family="sans-serif" font-size="13" fill="#0C447C"><text x="178" y="90">w₁</text><text x="182" y="133">w₂</text><text x="178" y="188">w₃</text></g><g><circle cx="90" cy="68" r="22" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.8"/><circle cx="90" cy="140" r="22" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.8"/><circle cx="90" cy="212" r="22" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.8"/></g><g font-family="sans-serif" font-size="14" fill="#0C447C" text-anchor="middle"><text x="90" y="73">x₁</text><text x="90" y="145">x₂</text><text x="90" y="217">x₃</text></g><circle cx="300" cy="140" r="40" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2"/><text x="300" y="137" text-anchor="middle" font-family="sans-serif" font-size="24" fill="#5A3E14">Σ</text><text x="300" y="157" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#5A3E14">+ b</text><circle cx="300" cy="250" r="18" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.6"/><text x="300" y="255" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#5A3E14">b</text><text x="367" y="131" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#33312E">z</text><circle cx="430" cy="140" r="34" fill="#D7EFE5" stroke="#1D9E75" stroke-width="2"/><text x="430" y="148" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#0E5E45">φ</text><text x="538" y="147" font-family="sans-serif" font-size="20" fill="#33312E">a</text><g font-family="sans-serif" font-size="11" fill="#9C9B95" text-anchor="middle"><text x="300" y="205">weighted sum (linear)</text><text x="430" y="192">activation (nonlinear)</text></g></svg></div>

The two stages have very different jobs: the **linear** part ($z$) *mixes* the
inputs; the **nonlinear** part ($\varphi$) *decides / squashes*. Essentially all
of deep learning is alternating these two operations.

A single neuron goes by several names: a **unit** or **node**, and — with a hard
step activation — a **perceptron** or **threshold logic unit (TLU)**.

**A worked example.** Take $\mathbf w=(0.5,\,-1,\,2)$, $b=0.5$, input
$\mathbf x=(2,\,1,\,0)$, and a sigmoid $\varphi$:
$$z = (0.5)(2) + (-1)(1) + (2)(0) + 0.5 = 0.5,\qquad a=\sigma(0.5)=\frac{1}{1+e^{-0.5}}\approx 0.62.$$
Now push $x_3$ from $0\to1$: $z$ jumps by $w_3=2$ to $2.5$, so $a=\sigma(2.5)\approx0.92$.
That "$z$ moves by $w_i$ when $x_i$ moves by 1" is exactly the meaning of a weight
(§2). Reproduce both numbers in the Playground with $n=3$.

**Vectorized / batched.** For a layer of $u$ neurons, stack the weight vectors
into $W\in\mathbb{R}^{u\times n}$ and biases into $\mathbf b\in\mathbb{R}^{u}$:
$\mathbf z = W\mathbf x + \mathbf b$. For a batch of $m$ inputs
$X\in\mathbb{R}^{m\times n}$: $Z = XW^{\top}+\mathbf b$ — one matrix multiply
(why GPUs love neural nets).

## 2. The three ingredients

A neuron has exactly **three kinds of knobs**, and each does one clean job. On a
1-input neuron you can see two of them directly: the **weight sets how steep** the
switch is, the **bias slides it left or right** (the activation, §5, sets the curve's
*shape*):

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 530 285" style="width:100%;max-width:530px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two panels. Left: a larger weight makes the sigmoid switch steeper. Right: the bias slides the same switch left or right."><rect x="1" y="1" width="528" height="283" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="55" y1="230" x2="245" y2="230" stroke="#D9D8D1" stroke-width="1"/><line x1="150" y1="62" x2="150" y2="240" stroke="#D9D8D1" stroke-width="1"/><line x1="55" y1="150" x2="245" y2="150" stroke="#ECEBE4" stroke-width="1" stroke-dasharray="3 3"/><polyline points="55,226 61,225 68,223 74,221 80,219 87,217 93,213 99,210 106,205 112,199 118,193 125,186 131,178 137,169 144,160 150,150 156,140 163,131 169,122 175,114 182,107 188,101 194,95 201,90 207,87 213,83 220,81 226,79 232,77 239,75 245,74" fill="none" stroke="#BFD8EF" stroke-width="2.5"/><polyline points="55,230 61,230 68,230 74,230 80,230 87,230 93,229 99,229 106,228 112,226 118,222 125,217 131,207 137,193 144,173 150,150 156,127 163,107 169,93 175,83 182,78 188,74 194,72 201,71 207,71 213,70 220,70 226,70 232,70 239,70 245,70" fill="none" stroke="#5B8FC2" stroke-width="2.5"/><polyline points="55,230 61,230 68,230 74,230 80,230 87,230 93,230 99,230 106,230 112,230 118,230 125,230 131,230 137,227 144,211 150,150 156,89 163,73 169,70 175,70 182,70 188,70 194,70 201,70 207,70 213,70 220,70 226,70 232,70 239,70 245,70" fill="none" stroke="#0C447C" stroke-width="2.5"/><line x1="285" y1="230" x2="475" y2="230" stroke="#D9D8D1" stroke-width="1"/><line x1="380" y1="62" x2="380" y2="240" stroke="#D9D8D1" stroke-width="1"/><line x1="285" y1="150" x2="475" y2="150" stroke="#ECEBE4" stroke-width="1" stroke-dasharray="3 3"/><polyline points="285,229 291,228 298,225 304,221 310,215 317,203 323,186 329,163 336,137 342,114 348,97 355,85 361,79 367,75 374,72 380,71 386,71 393,70 399,70 405,70 412,70 418,70 424,70 431,70 437,70 443,70 450,70 456,70 462,70 469,70 475,70" fill="none" stroke="#E6A4C0" stroke-width="2.5"/><polyline points="285,230 291,230 298,230 304,230 310,230 317,230 323,229 329,229 336,228 342,227 348,224 355,219 361,210 367,195 374,175 380,150 386,125 393,105 399,90 405,81 412,76 418,73 424,72 431,71 437,71 443,70 450,70 456,70 462,70 469,70 475,70" fill="none" stroke="#C0507A" stroke-width="2.5"/><polyline points="285,230 291,230 298,230 304,230 310,230 317,230 323,230 329,230 336,230 342,230 348,230 355,230 361,230 367,230 374,229 380,229 386,228 393,225 399,221 405,215 412,203 418,186 424,163 431,137 437,114 443,97 450,85 456,79 462,75 469,72 475,71" fill="none" stroke="#8A2351" stroke-width="2.5"/><g font-family="sans-serif"><text x="150" y="268" text-anchor="middle" font-size="13" fill="#0C447C">weight → steepness</text><text x="380" y="268" text-anchor="middle" font-size="13" fill="#8A2351">bias → position</text><text x="150" y="52" text-anchor="middle" font-size="11" fill="#9C9B95">small |w| … large |w|</text><text x="380" y="52" text-anchor="middle" font-size="11" fill="#9C9B95">slides left … right</text></g></svg></div>

**Weights $\mathbf w$** — one per input. The **sign** says excite ($+$) or inhibit
($-$); the **magnitude** says how much that input matters. Precisely,
$w_i=\partial z/\partial x_i$ is the neuron's **sensitivity** to input $i$ — nudge
$x_i$ a little and $z$ moves by $w_i$. So bigger $|w_i|$ ⇒ a steeper, more decisive
response (left panel), and scaling **all** weights up sharpens the switch toward a hard
step. Weights are what the network *learns*.

**Bias $b$** — the offset, equivalently a **threshold**. The neuron leans positive when
$z=\mathbf w\cdot\mathbf x + b \ge 0$, i.e. when the evidence clears the bar
$\mathbf w\cdot\mathbf x \ge -b$; so $\theta=-b$ *is* that bar. The bias lets the output
be nonzero when all inputs are zero, and **slides** the whole response curve left/right
(right panel) without changing its shape. *Bias trick:* append a constant input
$x_0=1$ with weight $w_0=b$; then $z=\tilde{\mathbf w}\cdot\tilde{\mathbf x}$ — bias is
"just another weight". (So one neuron has $n+1$ **parameters**: $n$ weights $+$ 1 bias.)

**Activation $\varphi$** — squashes/thresholds $z$ into the output. Its shape decides
whether the neuron behaves like a hard switch, a soft probability, or a rectifier (§5).

> **One line to remember:** *weights rotate & steepen the decision, the bias shifts it,
> the activation shapes the output.*

## 3. Four ways to picture one neuron

The same little formula $a=\varphi(\mathbf w\cdot\mathbf x+b)$ can be *seen* in four
ways — each makes a different thing click.

**(a) A weighted vote.** Picture each input casting a vote. Its **weight** says how
loud that vote is and which way it leans (a positive weight = "vote yes", negative =
"vote no"). The neuron **adds up the votes**; the **bias** is a head-start in one
direction. If the total clears the bar, the neuron says **yes (1)**, otherwise
**no (0)**. *Example — "should I go for a run?":* sunny ($+2$), it's raining ($-3$),
I have time ($+1$) → add them up; if the total is positive, go.

**(b) A line that splits space — the hyperplane.** *(the important one)*

Plot your inputs as points on a graph. A single neuron draws **one straight line** and
decides: everything on one side → **1**, everything on the other → **0**. That line is
the whole personality of the neuron:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 600 340" style="width:100%;max-width:560px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A neuron's straight decision line splits the plane into a 1 side and a 0 side; the weight vector points to the 1 side at right angles, and distance from the line is confidence."><defs><marker id="hpah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9A6A2A"/></marker></defs><rect x="1" y="1" width="598" height="338" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><polygon points="40,270 560,110 560,40 40,40" fill="#E6F1FB"/><polygon points="40,270 560,110 560,300 40,300" fill="#FBEAF0"/><line x1="40" y1="270" x2="560" y2="110" stroke="#33312E" stroke-width="2.5"/><line x1="300" y1="190" x2="275" y2="109" stroke="#9A6A2A" stroke-width="2.5" marker-end="url(#hpah)"/><line x1="210" y1="252" x2="189" y2="183" stroke="#6B6A66" stroke-width="1.4" stroke-dasharray="4 3"/><circle cx="210" cy="252" r="5" fill="#33312E"/><text x="300" y="326" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95">x₀</text><text x="20" y="172" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#9C9B95" transform="rotate(-90 20 172)">x₁</text><text x="450" y="80" font-family="sans-serif" font-size="15" fill="#0C447C">output = 1</text><text x="95" y="288" font-family="sans-serif" font-size="15" fill="#A32D2D">output = 0</text><text x="352" y="236" font-family="sans-serif" font-size="13" fill="#33312E">w·x + b = 0  (the line)</text><text x="247" y="100" font-family="sans-serif" font-size="15" fill="#9A6A2A">w</text><text x="150" y="270" font-family="sans-serif" font-size="11" fill="#6B6A66">distance = how sure</text></svg></div>

- The **line itself** is where the neuron is exactly undecided ($\mathbf w\cdot\mathbf x+b=0$).
- The **weights tilt / rotate** the line; the **bias slides** it across (try it in the Playground).
- The arrow $\mathbf w$ points straight at the **"1" side**, at right angles to the line —
  it's *the direction the neuron cares about*.
- **How far** a point sits from the line = **how sure** the neuron is. Right on the line = 50/50.

With **2 inputs** it's a **line**; with **3** a flat **plane**; with **many** we just call
it a **hyperplane** — same idea, more dimensions. So a neuron only ever asks two things:
**which side, and how far?**

**(c) A pattern detector.** The weights are the neuron's **favourite pattern**. Writing
the dot product as $\mathbf w\cdot\mathbf x=\lVert\mathbf w\rVert\,\lVert\mathbf x\rVert\cos\theta$
shows the output is **biggest when the input points the same way as the weights** (small
angle $\theta$) and **zero when they are at right angles**:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 360 210" style="width:100%;max-width:360px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The weight vector and the input vector with the angle theta between them; output is largest when they align."><defs><marker id="alah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#33312E"/></marker></defs><rect x="1" y="1" width="358" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="55" y1="175" x2="285" y2="50" stroke="#C0507A" stroke-width="2.5" marker-end="url(#alah)"/><line x1="55" y1="175" x2="245" y2="120" stroke="#185FA5" stroke-width="2.5" marker-end="url(#alah)"/><path d="M 110 147 A 60 60 0 0 0 99 137" fill="none" stroke="#6B6A66" stroke-width="1.4"/><text x="292" y="48" font-family="sans-serif" font-size="15" fill="#C0507A">w</text><text x="252" y="120" font-family="sans-serif" font-size="15" fill="#185FA5">x</text><text x="118" y="150" font-family="sans-serif" font-size="13" fill="#6B6A66">θ</text></svg></div>

So a neuron "lights up" for inputs that **look like its weights** — like a key matching a
lock. (Peek at the first layer of an image network and you literally see little edge and
colour templates.)

**(d) A soft logic gate.** With a steep activation and the right bias, the neuron behaves
like **AND / OR / NOT** — a fuzzy, adjustable yes/no switch. (Load the AND preset in the
Playground and slide the bias to turn it into OR.)

## 4. The physical picture

Map the math onto a (cartoon) physical neuron:

| symbol | physical reading |
|---|---|
| inputs $x_i$ | incoming signals — firing rates of upstream neurons / feature values |
| weights $w_i$ | **synaptic strengths / gains** ($+$ excitatory, $-$ inhibitory) |
| $\mathbf w\cdot\mathbf x$ | total synaptic drive — the "net input current" |
| bias $b$ | **excitability / threshold** — baseline readiness to fire |
| $z$ | "membrane potential" relative to threshold |
| $\varphi$ | the **f–I curve**: input current $\to$ output firing rate |
| $a$ | the neuron's firing rate / output signal |

The activation *is* the neuron's input–output response curve. A step = all-or-none
firing; a sigmoid = a saturating firing rate (there is a maximum rate); ReLU =
rate proportional to current above threshold, zero below. Weight *magnitude* is
gain; *learning* is adjusting synaptic strengths.

## 5. Activation functions — what they do and how they work

Three jobs, all essential:
1. **Inject nonlinearity** so depth means something (§8).
2. **Shape / bound the signal** — e.g. into a probability $(0,1)$ or a non-negative rate.
3. **Gate the gradient.** The derivative $\varphi'(z)$ is the neuron's **local
   gain** — how strongly a change in $z$ changes the output. Backprop multiplies
   these gains along every path, so $\varphi'$ decides whether the learning signal
   survives (§6).

The shapes at a glance — a hard **step**, the smooth **sigmoid**/**tanh** switches, and
the **ReLU** rectifier:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 480 300" style="width:100%;max-width:480px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Shapes of the step, sigmoid, tanh and ReLU activation functions on shared axes"><rect x="1" y="1" width="478" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="40" y1="200" x2="440" y2="200" stroke="#D9D8D1" stroke-width="1"/><line x1="240" y1="55" x2="240" y2="285" stroke="#D9D8D1" stroke-width="1"/><line x1="40" y1="130" x2="440" y2="130" stroke="#ECEBE4" stroke-width="1" stroke-dasharray="3 3"/><polyline points="40,200 240,200 240,130 440,130" fill="none" stroke="#9C9B95" stroke-width="2" stroke-dasharray="5 3"/><polyline points="40,200 53,200 67,200 80,199 93,199 107,199 120,198 133,197 147,196 160,194 173,192 187,188 200,184 213,178 227,172 240,165 253,158 267,152 280,146 293,142 307,138 320,136 333,134 347,133 360,132 373,131 387,131 400,131 413,130 427,130 440,130" fill="none" stroke="#185FA5" stroke-width="2.5"/><polyline points="40,270 53,270 67,270 80,270 93,270 107,270 120,270 133,270 147,269 160,269 173,267 187,265 200,258 213,246 227,227 240,200 253,173 267,154 280,142 293,135 307,133 320,131 333,131 347,130 360,130 373,130 387,130 400,130 413,130 427,130 440,130" fill="none" stroke="#7F77DD" stroke-width="2.5"/><polyline points="40,200 53,200 67,200 80,200 93,200 107,200 120,200 133,200 147,200 160,200 173,200 187,200 200,200 213,200 227,200 240,200 253,172 267,144 280,116 293,102 307,102 320,102 333,102 347,102 360,102 373,102 387,102 400,102 413,102 427,102 440,102" fill="none" stroke="#1D9E75" stroke-width="2.5"/><g font-family="sans-serif" font-size="11" fill="#6B6A66"><text x="446" y="134">1</text><text x="446" y="204">0</text><text x="446" y="274">−1</text><text x="432" y="295">z</text></g><g font-family="sans-serif" font-size="12"><text x="58" y="38" fill="#7d7c77">— step</text><text x="132" y="38" fill="#185FA5">— sigmoid</text><text x="240" y="38" fill="#7F77DD">— tanh</text><text x="322" y="38" fill="#1D9E75">— ReLU</text></g></svg></div>

**The catalogue** (switch between these in the Playground and watch the surface):

| $\varphi$ | formula | range | $\varphi'$ (gain) | character |
|---|---|---|---|---|
| step | $1$ if $z\ge0$ | $\{0,1\}$ | $0$ a.e. | hard switch; **no gradient → untrainable** by GD |
| sign | $\pm1$ | $\{-1,1\}$ | $0$ a.e. | bipolar step |
| linear | $z$ | $\mathbb{R}$ | $1$ | no squashing; regression output |
| sigmoid | $\dfrac{1}{1+e^{-z}}$ | $(0,1)$ | $\sigma(1-\sigma)\le0.25$ | smooth switch; **probability** |
| tanh | $\tanh z$ | $(-1,1)$ | $1-\tanh^2 z\le1$ | zero-centered sigmoid |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | $1$ if $z>0$ else $0$ | rectifier; sparse; cheap |
| Leaky/PReLU | $\max(\alpha z, z)$ | $\mathbb{R}$ | $1$ or $\alpha$ | fixes "dead" ReLU |
| ELU | $z$ or $\alpha(e^{z}-1)$ | $(-\alpha,\infty)$ | smooth | mean-centered, robust |
| GELU / SiLU | $z\,\Phi(z)$ / $z\,\sigma(z)$ | $\gtrsim-0.3$ | smooth | **Transformer default** |
| softmax | $e^{z_i}/\sum_j e^{z_j}$ | $(0,1)$, $\sum=1$ | Jacobian | **competition** across a layer |

**Physical / statistical meaning of the important ones**

- **Sigmoid = log-odds → probability.** If $a=\sigma(z)$ is a probability $p$,
  then $z=\ln\frac{p}{1-p}$ — the pre-activation *is the log-odds*, and each $w_i$
  is the "weight of evidence" that input contributes (same quantity as logistic
  regression / naive Bayes). Shape: the logistic / Fermi–Dirac curve — a smooth
  threshold that **saturates** toward 0 and 1.
- **tanh** is a rescaled sigmoid, $\tanh z = 2\sigma(2z)-1$, but **zero-centered**
  — outputs average near 0, keeping the next layer's inputs balanced and gradients
  better-conditioned. Still saturates.
- **ReLU = a half-wave rectifier (an ideal diode).** It passes positive current
  and blocks negative — exactly a diode, and a fair model of a neuron's
  non-negative firing rate. Consequences: **sparse** activations (many exact
  zeros), constant gain $1$ on the active side (**gradients don't shrink**), and
  trivial compute. Failure mode: a unit stuck at $z<0$ for all data has gain $0$
  forever — a **dead ReLU**. Leaky/PReLU/ELU give the negative side a small slope
  to avoid that.
- **GELU / SiLU (Swish)** are smooth ReLUs that *softly* gate each input by how
  positive it is ($x \times$ probability-it-is-positive). Smoothness helps
  optimization; they are the default in modern Transformers.
- **softmax is not pointwise — it is a competition.** It normalizes a *vector* of
  scores into a probability distribution, so raising one output lowers the others
  (winner-take-most). Physically it is the **Boltzmann distribution**:
  $p_i=\dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$, with $-z_i$ as energy and $T$ as
  **temperature**. $T\to0$ → argmax (one winner); large $T$ → uniform. (You meet
  this exact temperature again as the LLM sampling knob in Tier 5.)

## 6. Saturation, gain, and vanishing / exploding gradients

Backprop multiplies local gains $\varphi'$ along a path of depth $L$, so the
learning signal scales like a **product of gains**. With sigmoid
($\varphi'\le0.25$): through $L$ layers the gradient is at most $0.25^{L}$ — it
**vanishes** fast, and early layers barely learn. If gains are consistently $>1$,
gradients **explode**. ReLU's gain is exactly $1$ on the active side — which is
why ReLU-family activations + sensible initialization (He, e10) + normalization
made very deep nets trainable. **This is the single most important practical fact
about activations.**

Watch the gradient shrink with depth — sigmoid's best case ($0.25^L$) collapses toward
zero, while ReLU stays at 1:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 480 280" style="width:100%;max-width:480px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gradient size versus network depth: ReLU stays at 1 while the sigmoid product vanishes toward zero."><rect x="1" y="1" width="478" height="278" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><line x1="40" y1="50" x2="40" y2="240" stroke="#D9D8D1" stroke-width="1"/><line x1="40" y1="240" x2="450" y2="240" stroke="#D9D8D1" stroke-width="1"/><polyline points="40,50 440,50" fill="none" stroke="#1D9E75" stroke-width="2.5"/><polyline points="40,59 61,68 82,77 103,86 124,95 145,104 166,113 187,122 208,131 229,140 251,149 272,158 293,167 314,176 335,185 356,194 377,204 398,213 419,222 440,230" fill="none" stroke="#185FA5" stroke-width="2.5"/><g font-family="sans-serif" font-size="12"><text x="250" y="44" fill="#1D9E75">ReLU — gain stays 1</text><text x="232" y="150" fill="#185FA5">sigmoid — 0.25ᴸ vanishes</text></g><g font-family="sans-serif" font-size="10.5" fill="#9C9B95"><text x="34" y="54" text-anchor="end">1</text><text x="34" y="144" text-anchor="end">10⁻⁶</text><text x="34" y="234" text-anchor="end">10⁻¹²</text><text x="40" y="256" text-anchor="middle">1</text><text x="440" y="256" text-anchor="middle">20</text><text x="245" y="272" text-anchor="middle">network depth (layers) →</text></g></svg></div>

## 7. How to choose an activation

**Hidden layers**
- **Default: ReLU** — cheap, sparse, non-saturating. Start here.
- **Transformers / modern deep nets: GELU or SiLU.**
- **Dead units hurting you: Leaky ReLU or ELU.**
- **Avoid sigmoid/tanh in deep hidden stacks** — they saturate and vanish
  gradients (tanh is still fine shallow, and as RNN state).

**Output layer — pick by what you are predicting**

| task | output activation | paired loss |
|---|---|---|
| regression (real value) | **linear** (none) | MSE |
| positive value (count, price) | softplus / exp | MSE / Poisson |
| bounded value $[-1,1]$ | tanh | MSE |
| binary classification | **sigmoid** | binary cross-entropy |
| multiclass, one label | **softmax** | cross-entropy |
| multi-label, many labels | **sigmoid per class** | BCE per class |

**Two rules of thumb**
1. **Match activation to loss.** sigmoid+BCE and softmax+cross-entropy give the
   clean gradient $\partial L/\partial z = a-y$ (no $\varphi'$ factor to stall
   learning). Never pair softmax with MSE.
2. **Hidden = keep gradients alive (ReLU-family); output = match the target's
   range.**

(RNN cells use sigmoid for *gates* — values in $(0,1)$ act as soft on/off valves —
and tanh for the *cell state*.)

The whole decision in one picture — first ask *which layer*, then (for the output) *what
are you predicting*:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 660 300" style="width:100%;max-width:660px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A guide to choosing an activation. Ask which layer: hidden layers use ReLU by default, GELU or SiLU for transformers, Leaky ReLU or ELU for dead units, and avoid sigmoid or tanh deep. Output layers pick by task: linear for regression, sigmoid for binary, softmax for multiclass, sigmoid per class for multi-label — each paired with its matching loss."><defs><marker id="acg" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#9C9B95"/></marker></defs><rect x="1" y="1" width="658" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="262" y="16" width="136" height="36" rx="8" fill="#FFFFFF" stroke="#5B8FC2" stroke-width="1.6"/><text x="330" y="39" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#0C447C">Which layer?</text><line x1="288" y1="52" x2="178" y2="84" stroke="#9C9B95" stroke-width="1.6" marker-end="url(#acg)"/><line x1="372" y1="52" x2="486" y2="84" stroke="#9C9B95" stroke-width="1.6" marker-end="url(#acg)"/><text x="214" y="74" font-family="sans-serif" font-size="11" fill="#6B6A66">hidden</text><text x="430" y="74" font-family="sans-serif" font-size="11" fill="#6B6A66">output</text><rect x="28" y="88" width="288" height="150" rx="10" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.4"/><text x="172" y="112" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#0C447C">Hidden — keep gradients alive</text><g font-family="sans-serif" font-size="12" fill="#33312E"><text x="46" y="142">• ReLU — cheap default, start here</text><text x="46" y="168">• GELU / SiLU — transformers</text><text x="46" y="194">• Leaky ReLU / ELU — fix dead units</text><text x="46" y="220" fill="#A32D2D">✗ avoid sigmoid/tanh deep (vanish)</text></g><rect x="344" y="88" width="288" height="186" rx="10" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.4"/><text x="488" y="112" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#0E5E45">Output — match the target</text><g font-family="sans-serif" font-size="12" fill="#33312E"><text x="362" y="140">regression → linear  (MSE)</text><text x="362" y="164">binary → sigmoid  (BCE)</text><text x="362" y="188">multiclass → softmax  (cross-ent.)</text><text x="362" y="212">multi-label → sigmoid/class  (BCE)</text><text x="362" y="244" fill="#0E5E45">rule: pair activation with its loss</text></g></svg></div>

## 8. Why the nonlinearity is non-negotiable

If $\varphi$ were linear, $\varphi(z)=\alpha z+\beta$, a stack of neurons would
compute an affine function of an affine function — still **affine**. So *any*
depth collapses to one linear layer: zero extra power. The nonlinearity is what
lets a network bend space and carve curved decision regions. (XOR, §9, is the
smallest proof that one linear cut isn't enough.)

Picture a blue cluster ringed by red — no straight line can separate them no matter how
many *linear* layers you stack, but one nonlinearity lets the network bend the boundary
into a closed curve:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 520 250" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: a central blue cluster surrounded by a red ring cannot be split by any single straight line, which is all a stack of linear layers can produce. Right: a nonlinear network bends the boundary into a closed curve that cleanly encloses the blue cluster."><rect x="1" y="1" width="518" height="248" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="135" y="24" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">linear ∘ linear = one straight line</text><text x="385" y="24" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">+ nonlinearity = a curved boundary</text><line x1="42" y1="205" x2="235" y2="92" stroke="#33312E" stroke-width="2.4"/><g fill="#A32D2D"><circle cx="135" cy="67" r="4.5"/><circle cx="190" cy="95" r="4.5"/><circle cx="203" cy="140" r="4.5"/><circle cx="185" cy="188" r="4.5"/><circle cx="135" cy="203" r="4.5"/><circle cx="85" cy="188" r="4.5"/><circle cx="67" cy="140" r="4.5"/><circle cx="80" cy="95" r="4.5"/></g><g fill="#185FA5"><circle cx="135" cy="135" r="4.5"/><circle cx="118" cy="124" r="4.5"/><circle cx="152" cy="126" r="4.5"/><circle cx="124" cy="148" r="4.5"/><circle cx="148" cy="148" r="4.5"/><circle cx="135" cy="112" r="4.5"/></g><text x="135" y="236" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#A32D2D">no straight line separates them ✗</text><ellipse cx="385" cy="135" rx="48" ry="46" fill="none" stroke="#1D9E75" stroke-width="2.4" stroke-dasharray="6 3"/><g fill="#A32D2D"><circle cx="385" cy="67" r="4.5"/><circle cx="440" cy="95" r="4.5"/><circle cx="453" cy="140" r="4.5"/><circle cx="435" cy="188" r="4.5"/><circle cx="385" cy="203" r="4.5"/><circle cx="335" cy="188" r="4.5"/><circle cx="317" cy="140" r="4.5"/><circle cx="330" cy="95" r="4.5"/></g><g fill="#185FA5"><circle cx="385" cy="135" r="4.5"/><circle cx="368" cy="124" r="4.5"/><circle cx="402" cy="126" r="4.5"/><circle cx="374" cy="148" r="4.5"/><circle cx="398" cy="148" r="4.5"/><circle cx="385" cy="112" r="4.5"/></g><text x="385" y="236" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#1D9E75">a curved boundary encloses the cluster ✓</text></svg></div>

## 9. Logic gates & linear separability — the neuron's "hello world"

Why does the Playground ship with **AND, OR, NAND, XOR** presets? Because the
2-input boolean functions are the **smallest, exact, fully-visualizable tests** of
what one neuron can and cannot do — only 4 points, no dataset, and you can *see*
the decision line.

A single neuron is **one straight cut**. It computes a gate exactly when that gate
is **linearly separable** — one line with all the 1-outputs on one side, all the
0-outputs on the other.

- **AND, OR, NAND, NOR are linearly separable** → a single neuron *can* represent
  each (set the weights by hand in e01; the perceptron *learns* them in e02). In the
  Playground, load **AND** and slide only the **bias** — it morphs into **OR** (same
  line, shifted).
- **XOR and XNOR are NOT linearly separable** → **no single neuron can ever
  represent them**, with any weights: their two "1" corners sit on opposite
  diagonals, and no straight line separates them. This is the famous **Minsky &
  Papert (1969)** result that helped trigger the first "AI winter".

You can *see* it — one line cleanly splits AND, but no line can split XOR:

<div style="text-align:center;margin:0.7rem 0"><svg viewBox="0 0 520 285" style="width:100%;max-width:520px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AND is separable by one straight line; XOR is not — its two 1-corners sit on opposite diagonals."><rect x="1" y="1" width="518" height="283" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="55" y="70" width="160" height="160" fill="#FFFFFF" stroke="#D9D8D1"/><line x1="110" y1="63" x2="215" y2="177" stroke="#33312E" stroke-width="2"/><circle cx="80" cy="210" r="9" fill="#A32D2D"/><circle cx="190" cy="210" r="9" fill="#A32D2D"/><circle cx="80" cy="90" r="9" fill="#A32D2D"/><circle cx="190" cy="90" r="9" fill="#185FA5"/><rect x="305" y="70" width="160" height="160" fill="#FFFFFF" stroke="#D9D8D1"/><line x1="318" y1="120" x2="452" y2="180" stroke="#C7B8B8" stroke-width="1.6" stroke-dasharray="5 3"/><circle cx="330" cy="210" r="9" fill="#A32D2D"/><circle cx="440" cy="90" r="9" fill="#A32D2D"/><circle cx="330" cy="90" r="9" fill="#185FA5"/><circle cx="440" cy="210" r="9" fill="#185FA5"/><g font-family="sans-serif"><text x="135" y="258" text-anchor="middle" font-size="14" fill="#1D9E75">AND — one line works ✓</text><text x="385" y="258" text-anchor="middle" font-size="14" fill="#A32D2D">XOR — no line works ✗</text><text x="495" y="276" text-anchor="end" font-size="11" fill="#9C9B95">● = output 1   ● = output 0</text></g></svg></div>

**Why the gates matter (the point of the exercise):**
1. **They make "a neuron is a line" tangible** — and let you watch that line *fail*
   on XOR.
2. **XOR motivates depth.** The fix is a **hidden layer**: combine a few neurons
   (XOR = (A OR B) AND NOT(A AND B)) and the network bends space so XOR becomes
   separable. That one failure is *why* multilayer networks + backprop exist — the
   doorway to Tier 1 and beyond.
3. **Neurons can do logic at all.** McCulloch & Pitts (1943) first showed a neuron
   can act as a logic gate — the original bridge between brains and computation.
4. **NAND is functionally complete** — *any* boolean circuit can be built from NAND
   gates alone. So one neuron is a gate, and **layers of neurons can compute any
   logical function** (and far more): a neural net is, in principle, a universal
   computer.

Capacity footnote: the **VC dimension** of a linear classifier in $n$-D is $n+1$
(a 2D line shatters any 3 points in general position, but not 4) — and XOR's 4
points are exactly the case one line can't handle. Of the 16 two-input gates, **14
are linearly separable; only XOR and XNOR are not** (Cover, 1965).

## 10. It's the atom of classical ML too

| neuron | + loss / rule | = classical model |
|---|---|---|
| linear | mean squared error | **linear regression** |
| sigmoid | binary cross-entropy | **logistic regression** |
| sign | perceptron update | **perceptron** |
| linear | max-margin (hinge) | **linear SVM** |

So "single neuron" isn't a toy — it's logistic regression with a learnable threshold.

## 11. Cost, learning, and numerics

- **Cost:** one neuron is $O(n)$ multiply-adds; a layer on a batch is one matmul,
  $O(m\,n\,u)$; params $=n\cdot u+u$.
- **Learning the weights** (you set them by hand in e01): the **perceptron rule**
  (e02, converges iff linearly separable) or **gradient descent + backprop**
  (Tier 1, general).
- **Numerics:** naive $1/(1+e^{-z})$ overflows for very negative $z$ — the lab's
  `core/activations.py` splits by sign for stability. Weight **scale** matters:
  too big saturates $\varphi$, too small starves the signal; initialization (e10)
  controls this.

## 12. Actual neuron vs. artificial neuron — a "gross idealization"

An **actual (biological) neuron** receives electrochemical signals through its
**dendrites**, sums them in the **soma**, and — if the total crosses a threshold —
**fires** a spike down its **axon** to other neurons across **synapses** whose strengths
it can change. That last part is the whole game: *learning is changing synaptic strengths.*

An **artificial neuron** keeps exactly that skeleton and discards everything else — a
**gross idealization** of a brain cell, in Geoffrey Hinton's words, made *"so that we can
investigate how neurons can collaborate to do computations that we do not know how to
program."* We idealize **on purpose**: not to copy biology, but to get a unit simple enough
to analyze, differentiate, and stack by the billion.

| actual neuron | artificial neuron |
|---|---|
| dendrites (incoming signals) | inputs $x_i$ (from other neurons or sensors) |
| **synaptic strengths** | **weights** $w_i$ — what learning changes |
| soma sums the inputs | weighted sum $z=\sum_i w_i x_i + b$ |
| fires past a threshold | activation $\varphi(z)$ |
| axon spike to others | output $a$ — an input to the next layer |

Two things you'll use all lab long. A **ReLU** unit, $\varphi(z)=\max(0,z)$, is what Hinton
calls a **rectified linear neuron** — silent below threshold, linearly on above it (§5). And
**arranging these neurons in layers** — input → one or more *intermediate (hidden)* layers →
output — is "a common type of artificial neural network"; that stack is the MLP you train two
pages on, and each connection in it is one artificial "synapse" (a weight).

**The honest caveat:** real neurons **spike in time**, are stochastic, and have rich
dendritic dynamics the artificial unit ignores. The artificial neuron is a deliberate,
useful *caricature* — good enough to build intelligence from, not a faithful biological model.

## 13. Worked examples — one neuron in the wild

The same $a=\varphi(\mathbf w\cdot\mathbf x+b)$ already does useful work on its own
(it *is* logistic / linear regression). A few single-neuron models:

- **Loan approval** — $\mathbf x=[\text{income},\,\text{debt},\,\text{credit score}]$,
  sigmoid output = probability of repayment. Positive weights on income/credit,
  negative on debt; the bias sets how cautious the lender is.
- **Spam filter** — features like "contains the word *free*", "number of links",
  "ALL-CAPS ratio"; sigmoid output = $P(\text{spam})$. Each weight is a feature's
  "spamminess" — literally a log-odds (§5).
- **House price** — a *linear* neuron over $[\text{area},\,\text{bedrooms},\, \text{age}]$ → a price. (That's linear regression — see the ML module.)
- **Medical risk** — $[\text{age},\,\text{BMI},\,\text{glucose}]\to P(\text{diabetes})$.

**A tiny numeric trace (spam).** Weights $\mathbf w=[2.0,\,1.5,\,-1.0]$ for
[has "free", #links, from-known-contact], bias $b=-1$, input $\mathbf x=[1,\,3,\,0]$:
$$z = 2(1)+1.5(3)-1(0)-1 = 5.5,\qquad a=\sigma(5.5)\approx 0.996.$$
99.6% sure it's spam — the neuron *accumulated evidence*: "free" and the three
links pushed $z$ up, with no trusted-sender signal to pull it down.

## 14. From one neuron to today's AI

One neuron draws one boundary. The magic is **stacking**: a network of these units
with nonlinear activations is a **universal approximator** — it can represent
essentially any function. Add structure for the data type and you get every modern
system:

| add this structure | architecture | powers (real apps) |
|---|---|---|
| many layers (MLP) | deep network | tabular prediction, fraud, ranking |
| weight sharing over space | **CNN** | vision: face unlock, medical imaging, self-driving perception |
| recurrence / memory | RNN, LSTM | early speech & translation |
| **attention** | **Transformer** | **LLMs (Claude, GPT), translation, code (Copilot)** |
| learned vectors | embeddings | recommendation (Netflix, Spotify), semantic search, RAG |
| iterative denoising | diffusion | image generation (DALL·E, Midjourney, Stable Diffusion) |

**An LLM in one breath:** it is **billions of these exact neurons** in Transformer
blocks. Each block is mostly two things you already know — an **attention** step
(weighted sums deciding what to attend to) and an **MLP** (stacked neurons with a
**GELU** activation, §5). The final layer emits one score per vocabulary word; a
**softmax** (§5, with its **temperature** knob) turns those into next-token
probabilities; the model samples one, appends it, and repeats. Training is
**gradient descent + backprop** (§6) over trillions of words.

So nothing here is "toy". The neuron in the Playground, the activation you pick, the
softmax temperature, the match-loss-to-output rule — those are *literally* the parts
of the systems you use daily. This lab's arc (single neuron → MLP → attention → a
small GPT) is the real path from this panel to ChatGPT, just scaled up.
"""

_TASKS = r"""
### Warm-up — in the Playground tab
1. Load **AND (step)**, then drag **only the bias** from $-1.5$ toward $-0.5$.
   Watch it become **OR**. In one sentence: why does only $b$ change the gate?
2. Find $\mathbf w, b$ by hand for **NAND** and **NOR**; verify in the truth table.
3. Make a **sigmoid** neuron behave almost like a step: how large must
   $\lVert\mathbf w\rVert$ be so the four corner outputs are within $0.01$ of $0/1$?

### Pencil & paper
4. Write the boundary line for $\mathbf w=[1,2],\,b=-3$. Where does it cross each axis?
5. Prove XOR is not linearly separable (2 lines of argument).
6. Name the **2 of 16** two-input boolean functions a single neuron can't represent.
7. Derive the bias trick: show $\mathbf w\cdot\mathbf x+b=\tilde{\mathbf w}\cdot\tilde{\mathbf x}$.
8. Compute the signed distance from $\mathbf x=[2,0]$ to the boundary of
   $\mathbf w=[1,1],\,b=-1$.

### Activations
9. For each setting name the activation you'd pick and **why**: (a) hidden layer of
   a 30-layer net, (b) binary classifier output, (c) 10-class output, (d) bounded
   output in $[-1,1]$.
10. Show algebraically that $\tanh z = 2\sigma(2z)-1$ (so tanh is a rescaled sigmoid).
11. Confirm $\sigma'(z)$ peaks at $z=0$ with value $0.25$; then explain in one
    sentence why a 20-layer all-sigmoid net barely trains.

### Code — extend `core/`, keep tests green
12. Add `signed_distance(self, x)` to `core/neuron.Neuron` returning
    $(\mathbf w\cdot\mathbf x+b)/\lVert\mathbf w\rVert$; add a test in `tests/`.
13. Write a `boolean_fit(truth_table)` helper that brute-forces integer weights in
    $[-2,2]$ to realize a 2-input gate; confirm which gates are realizable.
14. **Bridge to e02:** implement the perceptron learning rule and train AND/OR
    from a random init.

### Stretch
15. Estimate capacity empirically: for $n=2$, randomly label $k$ points and measure
    the fraction a single neuron can separate as $k$ grows $2\to6$. Relate to
    VC dimension $=n+1=3$.
"""

_REFERENCES = r"""
### Foundational papers
- **McCulloch & Pitts (1943)** — *A Logical Calculus of the Ideas Immanent in
  Nervous Activity.* The first artificial neuron.
- **Rosenblatt (1958)** — *The Perceptron.* The first learning rule.
- **Minsky & Papert (1969)** — *Perceptrons.* The XOR / linear-separability limit.
- **Cover (1965)** — *Geometrical and Statistical Properties of Systems of Linear
  Inequalities* — capacity of linear separation.

### Books (with the right chapter)
- Nielsen — *Neural Networks and Deep Learning*, ch. 1 — [free online](http://neuralnetworksanddeeplearning.com/chap1.html).
- Goodfellow, Bengio & Courville — *Deep Learning*, ch. 6 — [deeplearningbook.org](https://www.deeplearningbook.org/).
- Bishop — *Pattern Recognition and ML*, ch. 4 (linear models for classification).
- Hastie, Tibshirani & Friedman — *Elements of Statistical Learning*, ch. 4.
- Hertz, Krogh & Palmer — *Introduction to the Theory of Neural Computation* (1991)
  — the classic graduate theory text (Hopfield nets, statistical mechanics).

### Courses & video
- **3Blue1Brown** — [neural networks series](https://www.3blue1brown.com/topics/neural-networks) (intuition).
- **Karpathy** — [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) + `micrograd`.
- **d2l.ai** — [Dive into Deep Learning](https://d2l.ai) (math + runnable code).

### In this lab
- Infobase: `infobase/00_foundations/single-neuron.md`, `activation-functions.md`,
  `perceptron.md`.
- Experiments: **e01** (this), **e02** (perceptron), **e03** (XOR fails).
- See `papers/README.md` for the full annotated reading list.
"""

_QUIZ = [
    Question(
        "In $z=\\mathbf{w}\\cdot\\mathbf{x}+b$, what does the bias $b$ do?",
        ["Scales the inputs", "Shifts the decision boundary (the threshold)",
         "Is the activation function", "Normalizes the output"],
        1,
        "b slides the hyperplane w·x+b=0; it's the threshold, equivalently a weight on a constant +1 input.",
    ),
    Question(
        "Why must a neuron include a nonlinear activation $\\varphi$?",
        ["To train faster", "To bound the weights",
         "Otherwise stacked neurons collapse to a single linear map",
         "Purely for biological realism"],
        2,
        "A composition of affine functions is affine, so without nonlinearity any depth has the power of one linear layer.",
    ),
    Question(
        "Geometrically, $\\mathbf{w}\\cdot\\mathbf{x}+b=0$ describes…",
        ["A single point", "A hyperplane separating two half-spaces",
         "A circle", "The activation curve"],
        1,
        "It's a hyperplane (line in 2D); w is its normal vector and it splits space into z>0 and z<0.",
    ),
    Question(
        "Given $\\mathbf{w}=[2,-1],\\,b=0.5,\\,\\mathbf{x}=[1,1]$, what is $z$?",
        ["0.5", "1.5", "2.5", "-0.5"],
        1,
        "z = 2·1 + (-1)·1 + 0.5 = 1.5.",
    ),
    Question(
        "With a step activation, which gate can a single neuron NOT compute?",
        ["AND", "OR", "XOR", "NAND"],
        2,
        "XOR is not linearly separable — no single line separates its classes (Minsky & Papert, 1969).",
    ),
    Question(
        "The maximum value of the sigmoid derivative $\\sigma'(z)$ is…",
        ["1", "0.5", "0.25 (at z=0)", "unbounded"],
        2,
        "At z=0, σ=0.5 so σ'=0.5·0.5=0.25. This small max is why deep sigmoid stacks vanish gradients.",
    ),
    Question(
        "A single sigmoid neuron trained with binary cross-entropy is equivalent to…",
        ["Linear regression", "Logistic regression", "k-nearest neighbors", "A decision tree"],
        1,
        "Sigmoid output + BCE loss is exactly logistic regression.",
    ),
    Question(
        "Increasing $\\lVert\\mathbf{w}\\rVert$ (same direction) makes a sigmoid neuron's transition…",
        ["Flatter", "Sharper, closer to a step", "Unchanged", "Inverted"],
        1,
        "Larger weights scale z, pushing outputs toward 0/1 — the boundary gets sharper. Try it in the Playground.",
    ),
    Question(
        "How is the bias absorbed into the weight vector (the 'bias trick')?",
        ["Add a constant input x₀=1 with weight w₀=b", "Multiply all weights by b",
         "Remove the activation", "Set b=0"],
        0,
        "With x̃=(1,x₁,…) and w̃=(b,w₁,…), z = w̃·x̃ — bias becomes just another weight.",
    ),
    Question(
        "The VC dimension (capacity) of a linear classifier in $n$-D input space is…",
        ["n", "n+1", "2ⁿ", "infinite"],
        1,
        "n+1: in 2D (n=2) a line can shatter 3 points but not 4.",
    ),
    Question(
        "Who introduced the perceptron learning rule?",
        ["McCulloch & Pitts", "Rosenblatt", "Minsky & Papert", "Rumelhart et al."],
        1,
        "Frank Rosenblatt, 1958. McCulloch & Pitts (1943) modeled the neuron; Rosenblatt added learning.",
    ),
    Question(
        "Writing $z=\\lVert\\mathbf{w}\\rVert\\,\\lVert\\mathbf{x}\\rVert\\cos\\theta$, an input of fixed "
        "length drives the neuron most strongly when it…",
        ["is orthogonal to w", "aligns in direction with the weight vector w",
         "points opposite to w", "has all-positive entries"],
        1,
        "z is maximal at cosθ=1, i.e. x parallel to w — the weight vector is the neuron's preferred stimulus / template.",
    ),
    Question(
        "Why is ReLU usually preferred over sigmoid in deep hidden layers?",
        ["It outputs probabilities", "It is zero-centered",
         "Its gain is 1 on the active side, so gradients don't vanish with depth",
         "It saturates on both sides"],
        2,
        "ReLU's derivative is 1 for z>0, so chained gains stay O(1); sigmoid's ≤0.25 gain shrinks as 0.25^L.",
    ),
    Question(
        "A 'dead' ReLU neuron is one that…",
        ["fires for every input", "has z<0 for all data, so its output and gradient are stuck at 0",
         "has exploding weights", "uses a negative bias only"],
        1,
        "If z<0 on all inputs, output=0 and gradient=0 forever — it can't recover. Leaky ReLU/ELU add a negative slope to prevent it.",
    ),
    Question(
        "In softmax $p_i=e^{z_i/T}/\\sum_j e^{z_j/T}$, increasing the temperature $T$…",
        ["makes the distribution sharper (closer to argmax)",
         "makes the distribution more uniform", "has no effect", "always picks the smallest score"],
        1,
        "Higher T flattens the distribution; T→0 approaches argmax. The same knob as LLM sampling temperature (Tier 5).",
    ),
    Question(
        "For single-label classification over 10 mutually exclusive classes, the output layer should use…",
        ["sigmoid per class + MSE", "softmax + cross-entropy", "tanh + MSE", "ReLU + BCE"],
        1,
        "Mutually-exclusive classes → softmax (a distribution summing to 1) paired with cross-entropy.",
    ),
    Question(
        "For a sigmoid neuron with output probability $p$, the pre-activation $z$ equals…",
        ["p itself", "the log-odds ln(p/(1−p))", "1−p", "p²"],
        1,
        "Inverting p=σ(z) gives z=ln(p/(1−p)) — the logit / log-odds; each weight is a 'weight of evidence'.",
    ),
    Question(
        "Stacking neurons with nonlinear activations can approximate…",
        ["only linear functions", "essentially any continuous function (universal approximation)",
         "only boolean functions", "nothing more than one neuron"],
        1,
        "The universal approximation theorem: a network with a hidden layer can approximate any continuous function given enough units.",
    ),
    Question(
        "An LLM turns its final-layer scores into next-token probabilities using…",
        ["ReLU", "softmax (with a temperature)", "a step function", "the raw bias term"],
        1,
        "Softmax over the vocabulary gives a probability per token; temperature controls how peaked the choice is.",
    ),
    Question(
        "Why does the lab use logic gates (AND, OR, XOR) to study a single neuron?",
        ["they are hard to compute",
         "they are the smallest exact, visualizable test of what one linear neuron can/can't represent",
         "they require large datasets", "they only run on GPUs"],
        1,
        "Four points, no dataset, a visible decision line — the perfect minimal probe of linear separability.",
    ),
    Question(
        "NAND gates are special in computing because…",
        ["they are the fastest gate",
         "they are functionally complete — any boolean circuit can be built from NANDs",
         "they need no inputs", "only brains can compute them"],
        1,
        "NAND is universal, so layered neurons (each able to act as a NAND) can in principle compute any logic.",
    ),
]

# --- Part 0: biological neuron — text ---------------------------------------

_BIO_INTRO = r"""
## 0. The biological neuron — where the idea comes from

Before any math: the artificial neuron is a deliberate *cartoon* of a real brain
cell. It throws away almost all the biology but keeps one thing — the
**computational role**. Seeing the real cell first makes every later symbol
($\mathbf w$, $b$, $\varphi$) feel inevitable.

Real neurons are **enormously complex and vary a lot** in their structure and
behaviour. So — like the field of **connectionism** — we focus on the **typical
properties shared by most neurons** and make the working assumption that *these*
account for the bulk of the brain's computing power. Here is that shared picture,
kept as simple as possible.
"""

_BIO_HOW = r"""
### How a real neuron works (and how signals travel)

A neuron has three working parts: **dendrites** (the inputs), the **soma** / cell
body (the integrator), and the **axon** (the output wire), which ends in **synapses**
onto other neurons. A single neuron may connect to **thousands** of others.

**At rest.** The inside of the cell sits slightly negative relative to the outside,
about **−70 mV** — like a tiny charged battery held ready to fire.

**Receiving signals.** Each incoming synapse gives the cell a small nudge:
**excitatory** inputs push the voltage **up** (toward firing), **inhibitory** inputs
push it **down**. The soma **adds up** all the nudges arriving together — this is
*integration over space*.

**The action potential (the "spike").** If the summed voltage crosses a **threshold**
(around **−55 mV**), the neuron **fires**: a sharp voltage spike — the **action
potential** — travels down the axon. It is **all-or-none**: a full spike or nothing,
like a switch, not a dimmer. Then the cell briefly resets to rest.

> *Bucket analogy:* picture a bucket filling with water. Excitatory inputs add water,
> inhibitory inputs drain it. When it overflows the rim (the threshold) it tips over
> (fires), empties, and starts refilling.

**How the signal travels — electrical, then chemical.** Along the **axon** the spike
is **electrical**, and a fatty **myelin** insulation makes it fast (up to ~120 m/s —
the gap between a quick reflex and a slow-arriving ache). At the **synapse** it turns
**chemical**: the arriving spike makes the axon tip release **neurotransmitter**
molecules across a tiny gap (the **synaptic cleft**); they cross and bind
**receptors** on the next neuron, nudging *its* voltage — and the cycle repeats. How
big that nudge is — the **synaptic strength** — is the key adjustable quantity (strong
vs. weak synapses), and **changing it is how the brain learns**.
"""

_BIO_KNOWLEDGE = r"""
### How connections store knowledge

The brain's knowledge is **not** in the neurons — it is in the **pattern and
strength of the connections** (synapses) between them. Three big ideas:

- **Synaptic strength = a weight.** A strong synapse passes a large signal, a weak
  one almost none — exactly the artificial weight $w_i$. The human brain has
  ~$10^{11}$ neurons and ~$10^{14}$–$10^{15}$ synapses: those synapses are the
  "parameters".
- **Hebbian learning — "neurons that fire together, wire together."** When neuron
  A reliably helps fire neuron B, the A→B synapse *strengthens* (long-term
  potentiation); uncorrelated activity *weakens* it. Learning = changing synaptic
  strengths — the biological ancestor of gradient descent updating weights.
- **Distributed representation.** A concept ("grandmother", "the letter A") is not
  one neuron — it is a *pattern of activity* across many neurons, stored as a
  *pattern of weights*. Each neuron takes part in many concepts. That is why the
  brain degrades gracefully when cells die, and why "knowledge" is smeared across
  the connections rather than filed in one place.
"""

_BIO_REASONING = r"""
### Where "reasoning" comes from

There is no central CPU. "Computation" is just **activity propagating through
weighted connections**:

- Sensory neurons fire → signals flow through layers of synapses → each downstream
  neuron fires if its *weighted* input crosses threshold → … → output neurons
  drive a response. **Inference = this forward propagation.**
- Recognizing a face, completing a pattern, recalling a memory from a partial cue
  are all the network *settling* into an activation pattern dictated by its
  weights. Recurrent loops let activity persist (working memory) and settle into
  stable **attractor** states — the idea behind Hopfield networks and associative
  memory.
- So: **memory = the weights; thinking = the dynamics of activation over those
  weights.** Change the weights (learn) and you change both what the system knows
  *and* how it reasons.
"""

_BIO_COMPARE = r"""
### Artificial vs. biological — the honest comparison

| biological | artificial | kept? |
|---|---|---|
| dendrite signals (firing rates) | inputs $x_i$ | kept (as real numbers) |
| synaptic strength | weight $w_i$ | kept |
| excitatory / inhibitory | $w_i>0$ / $w_i<0$ | kept |
| soma summation | $\sum w_i x_i$ | kept |
| firing threshold | bias $b$ | kept |
| firing-rate response (f–I curve) | activation $\varphi$ | kept (smoothed) |
| spike train in time | one number $a$ | **dropped** (no time) |
| Hebbian plasticity | gradient descent | replaced |
| ~$10^{15}$ synapses, ~20 watts | billions of params, kilowatts | very different |

**Kept:** integrate weighted inputs → compare to a threshold → emit a graded
output; learn by adjusting connection strengths. **Dropped:** spikes and timing,
ion-channel biophysics, dendritic computation, neuromodulators, and the brain's
astonishing energy efficiency.

A real neuron *inspired* the model, but the artificial neuron abstracts its
**computational job**, not the cell. The power of deep learning comes from
stacking millions of these caricatures and training them — not from biological
realism. (For the dynamics-of-activation view in depth, your 1994 course text —
Hertz, Krogh & Palmer — is still excellent.)
"""

# --- Part 0: biological neuron — figures (self-contained SVG) ----------------

_SVG_NEURON = """
<svg viewBox="0 0 760 300" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Anatomy of a biological neuron">
  <rect x="1" y="1" width="758" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <g stroke="#1D6FB8" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M180 150 L92 92 M126 116 L72 74 M126 116 L70 124"/>
    <path d="M178 150 L80 150 M112 150 L60 134 M112 150 L60 166"/>
    <path d="M180 150 L96 210 M128 188 L74 202 M128 188 L84 236"/>
  </g>
  <ellipse cx="215" cy="150" rx="48" ry="40" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2.5"/>
  <circle cx="215" cy="150" r="17" fill="#C98A3A" stroke="#8A5E22" stroke-width="2"/>
  <line x1="262" y1="150" x2="560" y2="150" stroke="#5B8FC2" stroke-width="3"/>
  <g fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2">
    <rect x="300" y="138" width="64" height="24" rx="12"/>
    <rect x="380" y="138" width="64" height="24" rx="12"/>
    <rect x="460" y="138" width="64" height="24" rx="12"/>
  </g>
  <g stroke="#1D9E75" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M560 150 L640 110 M560 150 L652 150 M560 150 L640 196"/>
  </g>
  <circle cx="645" cy="106" r="7" fill="#1D9E75"/>
  <circle cx="658" cy="150" r="7" fill="#1D9E75"/>
  <circle cx="645" cy="200" r="7" fill="#1D9E75"/>
  <g stroke="#6B6A66" stroke-width="1.5" fill="#6B6A66">
    <line x1="300" y1="272" x2="520" y2="272"/>
    <path d="M520 272 l-10 -5 l0 10 z"/>
  </g>
  <g fill="#33312E" font-family="sans-serif" font-size="13">
    <text x="58" y="52">dendrites (inputs)</text>
    <text x="180" y="236" text-anchor="middle">soma (cell body)</text>
    <text x="215" y="118" text-anchor="middle" font-size="11" fill="#6B6A66">nucleus</text>
    <text x="332" y="128" text-anchor="middle">myelin sheath</text>
    <text x="372" y="180" text-anchor="middle" font-size="11" fill="#6B6A66">node</text>
    <text x="430" y="200" text-anchor="middle">axon</text>
    <text x="648" y="90" text-anchor="middle">axon terminals</text>
    <text x="410" y="290" text-anchor="middle" font-size="11" fill="#6B6A66">signal flow</text>
  </g>
</svg>
"""

_SVG_SYNAPSE = """
<svg viewBox="0 0 760 320" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A synapse transmitting a signal">
  <rect x="1" y="1" width="758" height="318" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <ellipse cx="205" cy="175" rx="155" ry="95" fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2"/>
  <g fill="#7FB0E0" stroke="#5B8FC2" stroke-width="1.5">
    <circle cx="250" cy="140" r="12"/>
    <circle cx="272" cy="182" r="12"/>
    <circle cx="246" cy="216" r="12"/>
    <circle cx="305" cy="162" r="10"/>
  </g>
  <g fill="#C0507A">
    <circle cx="362" cy="150" r="5"/><circle cx="378" cy="172" r="5"/>
    <circle cx="366" cy="196" r="5"/><circle cx="388" cy="140" r="5"/>
    <circle cx="392" cy="186" r="5"/>
  </g>
  <rect x="418" y="70" width="55" height="210" rx="18" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2"/>
  <g fill="#9A6A2A">
    <rect x="408" y="128" width="14" height="16" rx="3"/>
    <rect x="408" y="165" width="14" height="16" rx="3"/>
    <rect x="408" y="202" width="14" height="16" rx="3"/>
  </g>
  <line x1="473" y1="175" x2="560" y2="175" stroke="#9A6A2A" stroke-width="3"/>
  <g>
    <text x="602" y="78" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">action potential</text>
    <line x1="540" y1="150" x2="668" y2="150" stroke="#B7B6B0" stroke-width="1"/>
    <polyline points="540,150 588,150 595,150 600,95 606,150 668,150" fill="none" stroke="#1D6FB8" stroke-width="2"/>
  </g>
  <g font-family="sans-serif" font-size="13" fill="#33312E">
    <text x="205" y="302" text-anchor="middle">presynaptic terminal</text>
    <text x="272" y="108" text-anchor="middle" font-size="11" fill="#6B6A66">vesicles</text>
    <text x="378" y="246" text-anchor="middle" font-size="11" fill="#C0507A">neurotransmitters</text>
    <text x="378" y="58" text-anchor="middle" font-size="11" fill="#6B6A66">cleft</text>
    <text x="505" y="302" text-anchor="middle">postsynaptic neuron</text>
    <text x="400" y="112" text-anchor="end" font-size="11" fill="#6B6A66">receptors</text>
  </g>
</svg>
"""

_SVG_MAP = """
<svg viewBox="0 0 760 270" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mapping between biological and artificial neuron parts">
  <rect x="1" y="1" width="758" height="268" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <text x="26" y="82" transform="rotate(-90 26 82)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#9A6A2A">biological</text>
  <text x="26" y="192" transform="rotate(-90 26 192)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1D6FB8">artificial</text>
  <g fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2">
    <rect x="100" y="58" width="100" height="44" rx="8"/>
    <rect x="235" y="58" width="100" height="44" rx="8"/>
    <rect x="370" y="58" width="100" height="44" rx="8"/>
    <rect x="505" y="58" width="100" height="44" rx="8"/>
    <rect x="640" y="58" width="100" height="44" rx="8"/>
  </g>
  <g fill="#CFE3F5" stroke="#5B8FC2" stroke-width="2">
    <rect x="100" y="168" width="100" height="44" rx="8"/>
    <rect x="235" y="168" width="100" height="44" rx="8"/>
    <rect x="370" y="168" width="100" height="44" rx="8"/>
    <rect x="505" y="168" width="100" height="44" rx="8"/>
    <rect x="640" y="168" width="100" height="44" rx="8"/>
  </g>
  <g font-family="sans-serif" font-size="12" fill="#5A3E14" text-anchor="middle">
    <text x="150" y="85">dendrites</text>
    <text x="285" y="85">synapses</text>
    <text x="420" y="85">soma sum</text>
    <text x="555" y="85">threshold</text>
    <text x="690" y="85">spike</text>
  </g>
  <g font-family="sans-serif" font-size="12" fill="#0C447C" text-anchor="middle">
    <text x="150" y="195">inputs xᵢ</text>
    <text x="285" y="195">weights wᵢ</text>
    <text x="420" y="195">Σ wᵢxᵢ + b</text>
    <text x="555" y="195">activation φ</text>
    <text x="690" y="195">output a</text>
  </g>
  <g stroke="#C8C7C0" stroke-width="1.5" stroke-dasharray="3 3">
    <line x1="150" y1="102" x2="150" y2="168"/>
    <line x1="285" y1="102" x2="285" y2="168"/>
    <line x1="420" y1="102" x2="420" y2="168"/>
    <line x1="555" y1="102" x2="555" y2="168"/>
    <line x1="690" y1="102" x2="690" y2="168"/>
  </g>
  <g fill="#9C9B95" font-family="sans-serif" font-size="16" text-anchor="middle">
    <text x="150" y="140">≈</text><text x="285" y="140">≈</text><text x="420" y="140">≈</text>
    <text x="555" y="140">≈</text><text x="690" y="140">≈</text>
  </g>
  <g stroke="#B0AFA8" stroke-width="1.5" fill="#B0AFA8">
    <line x1="200" y1="80" x2="231" y2="80"/><path d="M235 80 l-8 -4 l0 8 z"/>
    <line x1="335" y1="80" x2="366" y2="80"/><path d="M370 80 l-8 -4 l0 8 z"/>
    <line x1="470" y1="80" x2="501" y2="80"/><path d="M505 80 l-8 -4 l0 8 z"/>
    <line x1="605" y1="80" x2="636" y2="80"/><path d="M640 80 l-8 -4 l0 8 z"/>
    <line x1="200" y1="190" x2="231" y2="190"/><path d="M235 190 l-8 -4 l0 8 z"/>
    <line x1="335" y1="190" x2="366" y2="190"/><path d="M370 190 l-8 -4 l0 8 z"/>
    <line x1="470" y1="190" x2="501" y2="190"/><path d="M505 190 l-8 -4 l0 8 z"/>
    <line x1="605" y1="190" x2="636" y2="190"/><path d="M640 190 l-8 -4 l0 8 z"/>
  </g>
</svg>
"""

_BIO_TLU = r"""
### Modelling it: the artificial neuron (the TLU)

Now strip the biology down to what *computes*. The simplest model — the **Threshold
Logic Unit (TLU)**, drawn below — uses **two-valued signals**: every input and the
output is either **1** ("firing / spike") or **0** ("resting"). Each piece maps to one
real-neuron property:

- **Inputs $x_1\dots x_n$** — the incoming signals (0 or 1), i.e. whether each upstream
  neuron is firing.
- **Weights $w_i$** — each input is **multiplied by a weight** before it counts.
  *This is the synapse.* The **sign** of $w_i$ makes it excitatory ($w_i>0$, pushes
  toward firing) or inhibitory ($w_i<0$, pushes away); the **size** is the **synaptic
  strength** ($w_i=0$ means no connection).
- **SUM** — add the weighted inputs, $a=w_1x_1+\dots+w_nx_n$. This is the soma
  **integrating signals over space**.
- **Threshold $\theta$** — if the sum reaches $\theta$, output **$y=1$** (fire); else
  **$y=0$**. This is the **all-or-none action potential**, shown as the step on the right:

$$ y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \ge \theta \\ 0 & \text{otherwise} \end{cases} $$

(The **bias** from the math lessons is just the threshold moved across: $b=-\theta$, so
the rule becomes $\mathbf w\cdot\mathbf x + b \ge 0$.)

**A quick feel.** Inputs $x=[1,0,1]$, weights $w=[1,-2,1]$, threshold $\theta=1.5$:
sum $=1-0+1=2\ge1.5\Rightarrow$ **fires ($y=1$)**. Now switch the middle input on,
$x=[1,1,1]$: sum $=1-2+1=0<1.5\Rightarrow$ **silent ($y=0$)** — the inhibitory input
vetoed it. That is a neuron making a decision.
"""

_BIO_RATE = r"""
### From a single spike to a firing *rate* (and the sigmoid)

So far the signals took only **two values**. In a real neuron these are the **spike**
(the action-potential voltage, ≈ +30 mV — call it **"1"**) and the **resting** voltage
(≈ −70 mV — call it **"0"**). But a real neuron doesn't carry information in *one*
pulse: it encodes it in **how fast it fires** — its **firing rate** (spikes per
second). A weak stimulus → a few spikes per second; a strong one → a rapid train.
Different inputs produce different characteristic **firing patterns**.

How do we model a *rate* instead of a bare 0/1?

- **Hard threshold (TLU)** — output is only 0 or 1: fires or not. Simple, but it throws
  the rate away.
- **Continuous output (the *semilinear* neuron)** — let the output be a **number
  between 0 and 1** standing for the **firing rate**: 0 = silent, 1 = firing as fast as
  possible, 0.5 = half rate. The natural shape is the **sigmoid** — near the threshold
  the rate rises smoothly; far below it the neuron is silent (≈0); far above it
  **saturates** (≈1, because a neuron can't fire infinitely fast). **So the sigmoid is
  literally the neuron's firing-rate-vs-input curve** — the step is the idealised
  all-or-none switch, the sigmoid is the realistic graded version.
- **Stochastic version** — use that 0–1 number as the **probability** of emitting a
  spike at each instant. High value → likely to fire; low → unlikely. Averaged over
  time, that probability *is* the firing rate.

That's the concrete link behind the abstract "sigmoid activation": it's just a smooth
stand-in for **how often the neuron fires**.
"""

_BIO_SUMMARY = r"""
### The modelling summary

| real-neuron property | how the artificial neuron models it |
|---|---|
| integrate inputs over **space** (soma adds them) | a **weighted sum** $\sum_i w_i x_i$ |
| a **synapse** and its strength | **multiplication by a weight** $w_i$ (sign = excitatory / inhibitory) |
| the **action potential** (all-or-none) | a **threshold / step** → output 0 or 1 (the TLU) |
| the **firing rate** (how fast it spikes) | a **continuous output** via the **sigmoid** (or, stochastically, a spike *probability*) |
| integrate inputs over **time** | the **leaky-integrator** model — the cell gradually "forgets" old inputs |

And one property comes for free: like real brains, these neurons are **robust** — they
keep behaving sensibly when some inputs are noisy or missing, or some units fail,
because each decision rests on a **sum of many** contributions, not any single one.

Every modelling choice — sum, weight, threshold, sigmoid, leaky integrator — is a
deliberate, simplified stand-in for one property of the real cell: enough to compute
with, without the biophysics.
"""

_SVG_TLU = """
<svg viewBox="0 0 760 400" style="width:100%;height:auto;max-width:780px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Threshold Logic Unit: inputs times weights, summed, then thresholded to a 0/1 output">
  <defs>
    <marker id="tluah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#33312E"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="758" height="398" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/>
  <g stroke="#33312E" stroke-width="2" fill="none">
    <line x1="70" y1="70" x2="250" y2="158" marker-end="url(#tluah)"/>
    <line x1="55" y1="150" x2="244" y2="183" marker-end="url(#tluah)"/>
    <line x1="70" y1="330" x2="250" y2="232" marker-end="url(#tluah)"/>
  </g>
  <g stroke="#33312E" stroke-width="1.6" fill="#FFFFFF">
    <circle cx="180" cy="114" r="15"/><circle cx="160" cy="167" r="15"/><circle cx="180" cy="282" r="15"/>
  </g>
  <g stroke="#33312E" stroke-width="1.4">
    <line x1="170" y1="104" x2="190" y2="124"/><line x1="190" y1="104" x2="170" y2="124"/>
    <line x1="150" y1="157" x2="170" y2="177"/><line x1="170" y1="157" x2="150" y2="177"/>
    <line x1="170" y1="272" x2="190" y2="292"/><line x1="190" y1="272" x2="170" y2="292"/>
  </g>
  <circle cx="300" cy="195" r="58" fill="#EFD3AE" stroke="#9A6A2A" stroke-width="2.5"/>
  <text x="300" y="203" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#5A3E14">SUM</text>
  <line x1="358" y1="195" x2="442" y2="195" stroke="#33312E" stroke-width="2" marker-end="url(#tluah)"/>
  <rect x="445" y="132" width="215" height="126" rx="6" fill="#FFFFFF" stroke="#9C9B95"/>
  <line x1="478" y1="240" x2="478" y2="152" stroke="#6B6A66" stroke-width="1.2"/>
  <line x1="478" y1="240" x2="648" y2="240" stroke="#6B6A66" stroke-width="1.2"/>
  <line x1="565" y1="240" x2="565" y2="172" stroke="#B7B6B0" stroke-width="1" stroke-dasharray="3 3"/>
  <polyline points="478,240 565,240 565,172 642,172" fill="none" stroke="#185FA5" stroke-width="2.5"/>
  <text x="470" y="150" text-anchor="end" font-family="sans-serif" font-size="12" fill="#33312E">y</text>
  <text x="470" y="176" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6B6A66">1</text>
  <text x="652" y="244" font-family="sans-serif" font-size="12" fill="#33312E">a</text>
  <text x="565" y="254" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#33312E">θ</text>
  <line x1="660" y1="195" x2="712" y2="195" stroke="#33312E" stroke-width="2" marker-end="url(#tluah)"/>
  <text x="722" y="200" font-family="sans-serif" font-size="16" fill="#33312E">y</text>
  <g font-family="sans-serif" font-size="15" fill="#33312E">
    <text x="46" y="66">x₁</text><text x="31" y="146">x₂</text><text x="46" y="346">xₙ</text>
    <text x="198" y="104" font-size="13">w₁</text><text x="178" y="157" font-size="13">w₂</text><text x="198" y="272" font-size="13">wₙ</text>
  </g>
  <g fill="#6B6A66"><circle cx="118" cy="212" r="2.5"/><circle cx="126" cy="225" r="2.5"/><circle cx="134" cy="238" r="2.5"/></g>
  <rect x="430" y="300" width="300" height="76" rx="10" fill="#FAFAF7" stroke="#E2E2DA"/>
  <circle cx="472" cy="340" r="15" fill="#FFFFFF" stroke="#33312E" stroke-width="1.6"/>
  <g stroke="#33312E" stroke-width="1.4"><line x1="462" y1="330" x2="482" y2="350"/><line x1="482" y1="330" x2="462" y2="350"/></g>
  <text x="455" y="322" font-family="sans-serif" font-size="13" fill="#33312E">wᵢ</text>
  <text x="500" y="345" font-family="sans-serif" font-size="14" fill="#33312E">multiplication by wᵢ</text>
</svg>
"""

_BIO_BLOCKS = [
    {"md": _BIO_INTRO},
    {"svg": _SVG_NEURON,
     "caption": "A biological neuron: dendrites collect inputs, the soma integrates them, and the axon carries an all-or-none spike to other neurons."},
    {"md": _BIO_HOW},
    {"svg": _SVG_SYNAPSE,
     "caption": "The synapse. An arriving spike releases neurotransmitters across the cleft; receptors convert them into a voltage change in the next neuron. Synaptic strength = the size of that change = the 'weight'."},
    {"md": _BIO_KNOWLEDGE},
    {"md": _BIO_REASONING},
    {"svg": _SVG_MAP,
     "caption": "The correspondence. The artificial neuron keeps the computation — weighted sum, threshold, graded output, learning by changing connection strengths — and drops the biophysics and timing."},
    {"md": _BIO_TLU},
    {"svg": _SVG_TLU,
     "caption": "The Threshold Logic Unit: each input is multiplied by its weight (the synapse), the SUM integrates them (the soma), and the threshold θ turns the total into a 1/0 output (the all-or-none action potential)."},
    {"md": _BIO_RATE},
    {"md": _BIO_SUMMARY},
    {"md": _BIO_COMPARE},
]


SINGLE_NEURON = Lesson(
    key="single_neuron",
    title="The single artificial neuron",
    theory=_THEORY,
    quiz=_QUIZ,
    tasks=_TASKS,
    references=_REFERENCES,
    intro_blocks=_BIO_BLOCKS,
)

REGISTRY = {SINGLE_NEURON.key: SINGLE_NEURON}
