"""Post-training — from base LM to assistant (ANN module, roadmap e24–e26).

The Tiny-GPT / nanoGPT objective gives a **base** language model: it knows language and
facts but only *continues text*. Turning it into a helpful assistant (ChatGPT-style) takes
two more phases — **supervised fine-tuning (SFT)** on instruction demos, then **preference
tuning (RLHF / DPO)** on human rankings — done cheaply with **LoRA / PEFT**. This is a
concepts page (diagrams + theory), the alignment half of the story.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import streamlit as st

import lessons


_PIPELINE_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 720 210" style="width:100%;max-width:720px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The post-training pipeline: pretraining on web text gives a base LM that knows language; supervised fine-tuning on instruction demonstrations makes it follow instructions; preference tuning with RLHF or DPO on human rankings makes it helpful, harmless and honest."><defs><marker id="pt" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#5B8FC2"/></marker></defs><rect x="1" y="1" width="718" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><g font-family="sans-serif"><rect x="22" y="50" width="190" height="86" rx="10" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="1.6"/><rect x="266" y="50" width="190" height="86" rx="10" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.6"/><rect x="510" y="50" width="190" height="86" rx="10" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.6"/><g text-anchor="middle" font-size="13"><text x="117" y="74" fill="#0C447C">1. Pretraining</text><text x="361" y="74" fill="#5A3E14">2. SFT</text><text x="605" y="74" fill="#0E5E45">3. Preference tuning</text></g><g text-anchor="middle" font-size="10" fill="#6B6A66"><text x="117" y="96">next-token on web text</text><text x="117" y="112">→ knows language &amp; facts</text><text x="117" y="128" fill="#9C9B95">("base model")</text><text x="361" y="96">(instruction → ideal answer)</text><text x="361" y="112">→ follows instructions</text><text x="361" y="128" fill="#9C9B95">("instruct model")</text><text x="605" y="96">human rankings (RLHF / DPO)</text><text x="605" y="112">→ helpful, harmless, honest</text><text x="605" y="128" fill="#9C9B95">("aligned assistant")</text></g></g><g stroke="#5B8FC2" stroke-width="2" fill="none"><line x1="212" y1="93" x2="264" y2="93" marker-end="url(#pt)"/><line x1="456" y1="93" x2="508" y2="93" marker-end="url(#pt)"/></g><g font-family="sans-serif" font-size="10" fill="#9C9B95" text-anchor="middle"><text x="238" y="86">+demos</text><text x="482" y="86">+prefs</text></g><text x="360" y="176" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#6B6A66">huge data · little supervision → small, curated data · careful supervision (cost shrinks left → right)</text><text x="360" y="196" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#9A6A2A">SFT and preference tuning are usually done with LoRA — adapting ~0.1–1% of the weights</text></svg></div>'''

_LORA_SVG = '''<div style="text-align:center;margin:0.5rem 0"><svg viewBox="0 0 540 210" style="width:100%;max-width:540px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LoRA: keep the big pretrained weight matrix W frozen, and learn a small low-rank update B times A (with rank r much smaller than the dimension), so the effective weight is W plus B A. Only the tiny B and A are trained."><rect x="1" y="1" width="538" height="208" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><rect x="40" y="50" width="110" height="110" rx="6" fill="#E3E6EA" stroke="#9C9B95" stroke-width="1.6"/><text x="95" y="100" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#6B6A66">W</text><text x="95" y="120" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6B6A66">frozen</text><text x="95" y="178" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9C9B95">d × d (pretrained)</text><text x="172" y="110" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#33312E">+</text><rect x="196" y="50" width="34" height="110" rx="5" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.6"/><text x="213" y="110" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#5A3E14">B</text><text x="213" y="178" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9A6A2A">d × r</text><rect x="234" y="50" width="110" height="34" rx="5" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="1.6"/><text x="289" y="72" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#5A3E14">A</text><text x="289" y="100" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9A6A2A">r × d  (r ≪ d)</text><text x="372" y="110" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#33312E">=</text><rect x="398" y="50" width="120" height="110" rx="6" fill="#D7EFE5" stroke="#1D9E75" stroke-width="1.6"/><text x="458" y="104" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#0E5E45">W + BA</text><text x="458" y="124" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#0E5E45">adapted</text><text x="270" y="198" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="#9A6A2A">train only B and A (a few % of the weights) — W stays frozen</text></svg></div>'''


_THEORY = r"""
## 1. Two phases: pre-train, then align

A from-scratch GPT (Tiny GPT / e21) learns one thing: **predict the next token**. That makes
a **base model** that's fluent and knowledgeable — but it just *continues text*, it doesn't
*follow instructions*. Turning it into a ChatGPT-style **assistant** takes a second,
**post-training** stage. The full pipeline:

<PIPELINE/>

## 2. Pretraining — the base model

Next-token cross-entropy on a web-scale corpus (the e21 objective, scaled to trillions of
tokens). The model soaks up grammar, facts, styles, and reasoning patterns. Cost is
enormous and data is plentiful but **unsupervised**. Ask a base model "What is the capital
of France?" and it might *continue* with more questions rather than answer — it has the
knowledge but not the **behaviour**.

## 3. Transfer learning & fine-tuning

You rarely train from scratch. **Fine-tuning** starts from the pretrained weights (which
already encode useful features) and adapts them with **far less data** — the same idea as
using ImageNet features for a new vision task. Post-training is fine-tuning aimed at
*behaviour*, not new facts.

## 4. SFT — supervised fine-tuning (instruction tuning)

Fine-tune the base model on a curated set of **(instruction → ideal response)**
demonstrations written by humans. It learns to **follow instructions** and adopt a helpful
format/tone. Still plain next-token cross-entropy — just on demonstration data instead of
raw web text. Output: an **instruct model**.

## 5. Preference tuning — RLHF / DPO

Demonstrations don't capture *which of two good answers is better*. So we learn from
**human preferences**:
- **RLHF** — humans **rank** model outputs; train a **reward model** to predict those
  rankings; then optimize the LM (with **PPO**) to maximize reward, plus a **KL leash**
  keeping it close to the SFT model so it doesn't drift into gibberish:
  $$ \max_{\pi}\ \ \mathbb E\big[\,r(x,y)\,\big] \;-\; \beta\,\mathrm{KL}\!\big(\pi \,\|\, \pi_{\text{SFT}}\big). $$
- **DPO** (Direct Preference Optimization) — skip the separate reward model and PPO loop;
  optimize the preference objective **directly** with a simple classification-style loss.
  Simpler and now very common.

The result is an **aligned** model: more **helpful, harmless, and honest**.

## 6. LoRA / PEFT — adapt cheaply

Full fine-tuning updates **all** the weights — billions of numbers, huge memory. **LoRA**
(Low-Rank Adaptation) instead **freezes** the pretrained matrix $W$ and learns a small
**low-rank** update $\Delta W = BA$ (rank $r \ll d$), so the effective weight is $W + BA$:

<LORA/>

Only $B$ and $A$ are trained — often **~0.1–1%** of the parameters — which slashes memory and
lets you keep many small, **swappable adapters** for different tasks on top of one frozen
base. This is **PEFT** (parameter-efficient fine-tuning); SFT and preference tuning are
usually done this way.

## 7. RAG vs. fine-tuning (which problem?)

- **Need facts** — fresh, proprietary, citable? → **RAG** (Embeddings page): retrieve and
  put them in the prompt; no weight changes.
- **Need behaviour** — a format, tone, skill, or following instructions? → **fine-tuning**
  (SFT / preference tuning).

They compose: a typical assistant is **pretrained → SFT → preference-tuned**, and then uses
**RAG + tools** at inference for up-to-date facts.

## 8. Distillation — "dark knowledge" (Hinton, Vinyals & Dean, 2015)

Once you have a big, accurate **teacher** model, you can train a small, cheap **student** to
mimic it. The trick: don't train the student on the hard labels alone — train it on the
teacher's **full softmax distribution**, softened with a **temperature** $T$. Those soft
probabilities carry **"dark knowledge"** — the teacher's uncertainty says a *3* looks a bit
like an *8*, or that "delighted" sits near "happy" — information one-hot labels throw away.
The student matches the softened distribution (a cross-entropy / **KL** objective, Math
**X5**), and learns far faster and smaller than training from scratch. It's how giant models
are compressed to run on a phone, and conceptually the same bargain as LoRA: **keep the
knowledge, shrink the cost.**

## 9. The honest caveats

Post-training is mostly about **data quality** (good demos and clean preferences beat clever
losses), alignment is **ongoing** (jailbreaks, reward hacking, sycophancy are open problems),
and this is a **concepts** page — no training runs here. To actually train, scale the e21
nanoGPT and add an SFT/DPO step. 

## 10. The reward model, precisely

RLHF needs a scalar "how good is this response?" — but humans are unreliable at absolute
scores and quite reliable at **comparisons**. So annotators rank two responses, and a **reward
model** $r_\theta$ is trained on the **Bradley–Terry** likelihood that the preferred one wins:
$$ P(y_w \succ y_l \mid x)=\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big) $$
Maximising its log-likelihood is, once again, **logistic regression** (ML **M2**) — here over
*pairs* of responses. The reward model is usually the SFT model with its language-modelling
head swapped for a single scalar output.

## 11. PPO vs DPO — two ways to use those preferences

**PPO** (the original RLHF) treats generation as reinforcement learning: sample a response,
score it with the reward model, and push up the probability of high-reward text — while a
**KL penalty** to the SFT model stops it drifting:
$$ \max_\pi\; \mathbb E\big[r_\theta(x,y)\big]-\beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\text{SFT}}\big) $$
It works, but needs four models in memory (policy, reference, reward, critic) and is famously
fiddly.

**DPO** (2023) made a lovely observation: for that objective the *optimal* policy has a
closed form, so you can skip the reward model and the RL loop entirely and optimise the
preference pairs **directly** with a simple classification-style loss on the policy itself.
Same goal, one model, far more stable — which is why most open post-training now uses DPO or
a relative (IPO, KTO, ORPO, GRPO).

## 12. RLAIF and Constitutional AI

Human labels are the bottleneck: slow, expensive, inconsistent. **RLAIF** replaces the
labeller with a model — an AI critiques and ranks the responses. **Constitutional AI**
(Anthropic) makes this explicit: give the model a written set of principles (a
*constitution*), have it **critique and revise its own** output against them, train on the
revisions, and then use AI-generated preferences for the RL stage. The appeal is that the
values become **inspectable text** rather than being implicit in thousands of individual
annotations.

## 13. Chat templates — the format is part of the model

An instruct model is trained on a **specific** markup of roles, e.g.
`<|system|>…<|user|>…<|assistant|>…`. Those special tokens are how it knows who is speaking
and where to start writing. Two practical consequences: using the **wrong template** at
inference measurably degrades quality (the model is off-distribution), and the boundary
between "system instructions" and "user text" is only as strong as this training — which is
the root of **prompt injection**, since it is all just tokens in one stream.

## 14. What goes wrong

- **Reward hacking** — the policy finds inputs the reward model scores highly but humans
  dislike (excessive length, hedging, flattery). The KL penalty and better reward models are
  the defences.
- **Sycophancy** — annotators prefer agreeable answers, so the model learns to agree. A
  direct, measurable consequence of *who labelled the data*.
- **The alignment tax** — post-training can cost raw capability (a well-known effect on
  benchmarks); mixing pretraining data back in during SFT reduces it.
- **Mode collapse** — heavy RL narrows the output distribution: safer, more useful, less
  diverse and creative.

## 15. Evaluation — the hardest part

There is no loss function for "helpful". In practice: **human preference arenas** (pairwise
battles with Elo scores), **LLM-as-judge** (cheap and scalable, but biased toward length and
toward its own family's style), **static benchmarks** (MMLU, GSM8K — cheap, but they leak into
training data), and **red-teaming** for safety. Anyone claiming a single number captures
post-training quality is selling something.

## 16. Test-time compute — buying accuracy with inference, not parameters

Everything up to here spends compute **once**, at training time, and then answers in a
single forward pass. The largest change since is the discovery that you can also spend it
**at inference**, per question, and that the returns are as predictable as the scaling laws
from Level 4 — only the x-axis is now *tokens generated while answering* instead of
parameters trained.

### Why a scratchpad is not a stylistic choice

This is the part with real architecture content, and it connects straight back to
**Neurons compute**: a transformer of $L$ layers applies a **fixed** number of sequential
operations to produce one token. It is a wide combinational circuit with a hard depth
limit. No matter how large you make it, there are problems needing more sequential steps
than it has layers — and it cannot loop, because there is no feedback path.

Chain of thought removes that ceiling by an almost embarrassing trick: **write the
intermediate result into the context and read it back.** Each emitted token is another pass
through all $L$ layers, with everything written so far available, so the model gets

$$ \text{effective serial depth} \;\approx\; L \times (\text{tokens generated}) $$

The scratchpad *is* the recurrence the architecture does not have. That is why "think step
by step" works on multi-step arithmetic and fails to help on single-lookup recall — one is
depth-limited, the other is not. Read it as the transformer borrowing, at inference time,
exactly the capability the RNN had structurally and lost in the trade for parallel
training.

### The four ways to spend inference compute

| | what it does | cost | needs |
|---|---|---|---|
| **Chain of thought** | serial depth, as above | 1 longer answer | prompting alone |
| **Self-consistency** | sample $n$ chains, take the majority answer | $n\times$ | nothing |
| **Best-of-$n$** | sample $n$, let a **verifier** pick | $n\times$ + verifier | a reward model |
| **Long RL-trained reasoning** | the model learns *when* to keep thinking | variable | RLVR training |

The Decoding page covers greedy, temperature, top-k, top-p, beam and speculative decoding —
every strategy for producing *one* answer well. Self-consistency and best-of-$n$ are the
missing move: produce **many** and choose. Note the asymmetry that makes it work — checking
a candidate answer is usually far cheaper than producing it, the same reason NP-style
problems are easy to verify and hard to solve.

### RLVR — reward the answer, not the human's opinion

§10–§11 trained a reward model on human preferences, and warned it can be gamed. For any
task with a **checkable** answer, that whole apparatus is unnecessary: a maths problem has a
right answer, code either passes the tests or does not. So drop the learned reward model and
score the outcome directly:

$$ r \;=\; \mathbf{1}[\text{answer is correct}] $$

This is **reinforcement learning with verifiable rewards**, and it is a different safety
profile from RLHF, not just a cheaper one — a reward model can be fooled by an answer that
merely *looks* good, whereas a unit test cannot be flattered.

**GRPO** is the algorithm that made it practical. PPO (§11) needs a *critic* network
predicting expected return — roughly a second model to train and hold in memory. GRPO
deletes it: sample a **group** of $G$ answers to the same prompt, and use the group's own
mean reward as the baseline each answer is measured against:

$$ A_i \;=\; \frac{r_i - \operatorname{mean}(r_1,\dots ,r_G)}{\operatorname{std}(r_1,\dots ,r_G)} \qquad(\text{no critic anywhere}) $$

An answer better than its siblings gets reinforced, worse gets suppressed. The comparison
that PPO needed a learned critic for is obtained for free by generating several attempts —
the "baseline" is just the other answers to the same question. Half the memory, and one
fewer model that can be wrong.

### Outcome vs. process rewards

Scoring only the final answer (**ORM**) is cheap and honest but rewards a lucky guess
reached by bad reasoning. Scoring **each step** (**PRM**) gives far denser signal and
catches reasoning that arrives correctly by accident, but needs step-level labels, which are
expensive and themselves debatable. The pragmatic middle — outcome rewards for training,
process rewards as the *verifier* in best-of-$n$ — is where most published systems sit.

### The honest part

The cost is real and it is paid on **every** query, forever, unlike a training cost paid
once. A long reasoning trace can be tens of thousands of tokens for one answer, which is
why these models are metered differently and why "should this question get the expensive
path?" is now a product decision. And more thinking is not monotonically better: past some
length, accuracy flattens or falls as the model talks itself out of a correct early answer.

Two things worth holding onto. First, none of this is an architecture change — **the block
from Level 4 is untouched**; this is entirely training and inference procedure. Second, it
reopened a direction everyone assumed was closed: when parameter scaling got expensive, the
axis that was still cheap turned out to be *time spent thinking*, and it had been sitting
there since the first transformer.

*(Frontier labs publish little detail here; treat specific recipes as informed reconstruction.
See the References tab. Roadmap e24–e26.)*

## 17. Where the frontier is

Post-training used to be a thin polish on a big pretrain; it is now a large fraction of the
work — and increasingly *is* the product. Beyond reasoning (§16), the other live direction is
**tool use / agents** — training the model to call functions, search and execute code, so it
can act rather than just answer. Both are post-training problems, not architecture problems
— the block from Level 4 has not changed.

*(Roadmap e24–e26.)*
"""

_QUIZ = [
    lessons.Question(
        "A pretrained 'base' LM, before post-training, mainly…",
        ["follows instructions well", "continues text — it's fluent and knowledgeable but not instruction-following",
         "can't produce language", "is already aligned"], 1,
        "Next-token pretraining gives fluency + knowledge; instruction-following and alignment come later (SFT, preference tuning)."),
    lessons.Question(
        "Supervised fine-tuning (SFT) trains the model on…",
        ["random web text", "curated (instruction → ideal response) demonstrations",
         "human rankings only", "images"], 1,
        "SFT is next-token training on demonstration data, teaching the model to follow instructions."),
    lessons.Question(
        "RLHF / DPO use which signal?",
        ["the next token only", "human preferences (rankings of outputs)",
         "the learning rate", "the tokenizer"], 1,
        "Preference tuning aligns the model to human rankings — RLHF via a reward model + PPO, DPO directly."),
    lessons.Question(
        "LoRA makes fine-tuning cheap by…",
        ["using a bigger model", "freezing W and training only a small low-rank update B·A (~0.1–1% of params)",
         "removing layers", "skipping the gradient"], 1,
        "Low-rank adapters add W+BA with r≪d, so only B and A are trained — tiny, swappable."),
    lessons.Question(
        "Need the model to cite fresh, proprietary facts. Best tool?",
        ["fine-tuning", "RAG (retrieve facts into the prompt)", "a bigger learning rate", "dropout"], 1,
        "RAG supplies knowledge at query time; fine-tuning is for behaviour/style. They complement each other."),
    lessons.Question(
        "Why does a chain of thought give a transformer capability it structurally lacks?",
        ["It makes the model larger",
         "Each generated token is another pass through all L layers, so writing intermediate "
         "results into the context turns a fixed-depth circuit into a variable-depth one",
         "It changes the attention pattern",
         "It retrains the weights"], 1,
        "A transformer applies a fixed number of sequential ops per token and has no feedback "
        "path. The scratchpad is the recurrence the architecture gave up for parallel training "
        "(§16, and the Neurons compute page)."),
    lessons.Question(
        "What does GRPO remove compared with PPO, and what replaces it?",
        ["The reward model; replaced by human labels",
         "The critic network; replaced by the mean reward of a group of answers to the same prompt",
         "The policy gradient; replaced by supervised learning",
         "Nothing — they are the same algorithm"], 1,
        "PPO needs a learned critic for its baseline. Sampling G answers to one prompt gives a "
        "baseline for free, so the second network disappears (§16)."),
    lessons.Question(
        "RLVR replaces the learned reward model with a verifier. Why is that a safety "
        "improvement and not only a cost saving?",
        ["Verifiers are faster",
         "A reward model can be fooled by an answer that merely looks good; a unit test or a "
         "checked numeric answer cannot be flattered",
         "It removes the need for training data",
         "It prevents overfitting"], 1,
        "Reward hacking is the failure mode of §10's learned reward model. A verifiable reward "
        "has no opinion to game — though it only exists for checkable tasks."),
]

_TASKS = r"""
### Concept
1. In one line each, say what **pretraining**, **SFT**, and **preference tuning** add to the model.
2. Why does RLHF include a **KL penalty** keeping the model near the SFT model? What breaks
   without it?
3. Explain **LoRA** to someone who knows matrix multiply: what is frozen, what is trained,
   and why it's cheap.

### Decide
4. For each, pick **RAG** or **fine-tuning**: (a) answer from today's news; (b) always reply
   in your company's brand voice; (c) cite an internal policy doc; (d) reliably output valid JSON.

### Build (stretch)
5. Sketch how you'd add an **SFT** step to the e21 nanoGPT (data format, loss, what changes).
"""

_REFS = r"""
- Ouyang et al. (2022) — *InstructGPT* (SFT + RLHF, the ChatGPT recipe).
- Rafailov et al. (2023) — *Direct Preference Optimization (DPO)*.
- Hu et al. (2021) — *LoRA*; Dettmers et al. (2023) — *QLoRA*.
- Hinton, Vinyals & Dean (2015) — *Distilling the Knowledge in a Neural Network* ("dark knowledge").
- **Geoffrey Hinton** — *Neural Networks for Machine Learning* (his lectures on YouTube): the two
  paradigms of intelligence, the family-tree net, and much of this lab's framing come from here.
- Wei et al. (2022) — *Chain-of-Thought Prompting Elicits Reasoning in LLMs*.
- Wang et al. (2022) — *Self-Consistency Improves Chain of Thought Reasoning*.
- Cobbe et al. (2021) — *Training Verifiers to Solve Math Word Problems* (best-of-n, GSM8K).
- Lightman et al. (2023) — *Let's Verify Step by Step* (process vs outcome reward models).
- Shao et al. (2024) — *DeepSeekMath* (**GRPO**: the group baseline that removes the critic).
- DeepSeek-AI (2025) — *DeepSeek-R1* (RL with verifiable rewards at scale, openly described).
- Snell et al. (2024) — *Scaling LLM Test-Time Compute Optimally*.
- Merrill & Sabharwal (2024) — *The Expressive Power of Transformers with Chain of Thought*
  (the formal version of the serial-depth argument in §16).
- Hugging Face — *TRL* (SFT/DPO/PPO) and *PEFT* libraries.
- In this lab: **Tiny GPT** / **e21 nanoGPT** (pretraining), **Embeddings & RAG**
  (facts vs behaviour), Math **X5** (cross-entropy / KL).
"""


st.title("Post-training — from base LM to assistant")
st.caption("A from-scratch GPT only continues text. SFT + preference tuning (RLHF / DPO), "
           "done cheaply with LoRA, turn it into a helpful assistant. (Concepts + diagrams.)")

lessons.predict(
    'A from-scratch GPT can only *continue* text. What two stages turn it into a helpful **assistant** that follows instructions?',
    '**SFT** (supervised fine-tuning on instruction→response pairs) teaches the format, then **preference tuning** (RLHF / DPO) aligns *which* response is preferred — done cheaply with **LoRA** (train small adapters, freeze the base). The knowledge is mostly already in the base model; these stages shape *behavior*.',
)

tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_theory:
    st.markdown(_THEORY.replace("<PIPELINE/>", _PIPELINE_SVG).replace("<LORA/>", _LORA_SVG),
                unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(_QUIZ, prefix="posttrain")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(_TASKS)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** **Pretraining**: broad next-token knowledge from raw text. **SFT**: teaches instruction-following *format* from curated prompt→response pairs. **Preference tuning** (RLHF/DPO): aligns *which* response is preferred (helpful, harmless).

**2.** The **KL penalty** keeps the tuned model close to the SFT model so it can't "reward-hack" — drift into degenerate, high-reward-model-scoring gibberish. Without it, fluency and diversity collapse as the policy games the reward.

**3.** **LoRA**: freeze the big pretrained matrix $W$ and add a low-rank update $\Delta W = A B$ with $A\!\in\!\mathbb{R}^{d\times r}$, $B\!\in\!\mathbb{R}^{r\times d}$, $r\ll d$. Only $A,B$ train — orders of magnitude fewer parameters, cheap memory, and you can hot-swap adapters.""",
        label="Concept 1–3",
    )
    lessons.solution(
        r"""**4.** (a) today's news → **RAG** (freshness, no retrain). (b) brand voice → **fine-tuning** (a behavior/style). (c) cite an internal policy doc → **RAG** (retrieve + ground + cite). (d) reliably valid JSON → **fine-tuning** (a consistent output format).""",
        label="Decide 4",
    )
    lessons.solution(
        r"""**5.** Add SFT to e21: feed **(prompt, response)** pairs instead of raw text, and **mask the loss to the response tokens only** (don't train on predicting the prompt). Same cross-entropy objective, same architecture — only the data and the loss mask change.""",
        label="Build (stretch) 5",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(_REFS)
