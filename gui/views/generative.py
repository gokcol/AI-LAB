"""The generative family — autoregressive, VAE, GAN, diffusion (ANN module).

Placed after Tiny GPT because that is where the question lands: you have just trained a
model that invents text, so what *kind* of thing is it, and what are the others? All four
attack the same problem — learn p(x) well enough to draw new samples from it — and they
differ almost entirely in how they dodge the fact that p(x) is intractable to write down.

The Live tab is deliberately 1-D and tiny: every one of these families can be demonstrated
on a two-bump distribution in a few hundred numbers, and seeing autoregression and
diffusion produce the *same* target from opposite directions is the point of the page.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons

st.title("The generative family — four ways to learn $p(x)$")
st.caption("Autoregressive · VAE · GAN · Diffusion — what each optimises, and why the "
           "winner is different for text and for images.")

# --------------------------------------------------------------------------- #
# Diagram
# --------------------------------------------------------------------------- #
_MAP_SVG = '''<div style="text-align:center;margin:0.4rem 0 1rem"><svg viewBox="0 0 780 300" style="width:100%;max-width:800px;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Four families of generative model. Autoregressive models factorise the joint probability into a product of conditionals and maximise exact likelihood. Variational autoencoders encode to a latent distribution and maximise a lower bound on the likelihood. Generative adversarial networks train a generator against a discriminator with no likelihood at all. Diffusion models add noise over many steps and learn to reverse it, maximising a bound on the likelihood."><defs><marker id="gm" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#7A8A99"/></marker></defs><rect x="1" y="1" width="778" height="298" rx="14" fill="#FAFAF7" stroke="#E2E2DA"/><text x="390" y="26" text-anchor="middle" font-family="sans-serif" font-size="12.5" fill="#6B6A66">every family wants the same thing — samples from <tspan font-style="italic">p(x)</tspan> — and differs in what it is willing to give up</text><g font-family="sans-serif"><rect x="24" y="48" width="170" height="118" rx="10" fill="#E6F1FB" stroke="#5B8FC2" stroke-width="2"/><text x="109" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#0C447C">Autoregressive</text><text x="109" y="92" text-anchor="middle" font-size="10.5" fill="#33556F">factorise exactly</text><text x="109" y="112" text-anchor="middle" font-size="11" fill="#0C447C">&#8719; p(x&#8348; | x&lt;&#8348;)</text><text x="109" y="134" text-anchor="middle" font-size="9.5" fill="#5B7C99">exact likelihood</text><text x="109" y="150" text-anchor="middle" font-size="9.5" fill="#5B7C99">slow sampling</text><rect x="212" y="48" width="170" height="118" rx="10" fill="#DCEFE2" stroke="#1D9E75" stroke-width="2"/><text x="297" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#0E5E45">VAE</text><text x="297" y="92" text-anchor="middle" font-size="10.5" fill="#2C6B54">encode to a latent</text><text x="297" y="112" text-anchor="middle" font-size="11" fill="#0E5E45">x &#8594; z &#8594; x&#770;</text><text x="297" y="134" text-anchor="middle" font-size="9.5" fill="#5E8E76">lower bound (ELBO)</text><text x="297" y="150" text-anchor="middle" font-size="9.5" fill="#5E8E76">blurry samples</text><rect x="400" y="48" width="170" height="118" rx="10" fill="#FBEAF0" stroke="#C0507A" stroke-width="2"/><text x="485" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#8A2351">GAN</text><text x="485" y="92" text-anchor="middle" font-size="10.5" fill="#A33D66">fool a critic</text><text x="485" y="112" text-anchor="middle" font-size="11" fill="#8A2351">G vs D</text><text x="485" y="134" text-anchor="middle" font-size="9.5" fill="#B06A87">no likelihood at all</text><text x="485" y="150" text-anchor="middle" font-size="9.5" fill="#B06A87">sharp, unstable</text><rect x="588" y="48" width="170" height="118" rx="10" fill="#FBEAD6" stroke="#9A6A2A" stroke-width="2"/><text x="673" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#5A3E14">Diffusion</text><text x="673" y="92" text-anchor="middle" font-size="10.5" fill="#7A5620">destroy, then undo</text><text x="673" y="112" text-anchor="middle" font-size="11" fill="#5A3E14">x &#8594; noise &#8594; x</text><text x="673" y="134" text-anchor="middle" font-size="9.5" fill="#9A7B4F">bound, stable</text><text x="673" y="150" text-anchor="middle" font-size="9.5" fill="#9A7B4F">many steps</text></g><g font-family="sans-serif" font-size="10.5"><rect x="24" y="196" width="358" height="80" rx="9" fill="#EEF4FB" stroke="#C4D6E8"/><text x="203" y="217" text-anchor="middle" font-weight="700" fill="#0C447C">text &#8594; autoregressive wins</text><text x="203" y="237" text-anchor="middle" fill="#33556F">discrete, ordered, variable length —</text><text x="203" y="253" text-anchor="middle" fill="#33556F">left-to-right is how language already is,</text><text x="203" y="268" text-anchor="middle" fill="#33556F">and the factorisation costs nothing</text><rect x="400" y="196" width="358" height="80" rx="9" fill="#FBF3E6" stroke="#E3D2B4"/><text x="579" y="217" text-anchor="middle" font-weight="700" fill="#5A3E14">images &#8594; diffusion wins</text><text x="579" y="237" text-anchor="middle" fill="#7A5620">continuous, no natural order —</text><text x="579" y="253" text-anchor="middle" fill="#7A5620">refining the whole canvas at once beats</text><text x="579" y="268" text-anchor="middle" fill="#7A5620">inventing pixels one raster line at a time</text></g></svg></div>'''

_TARGET = "two bumps"


@st.cache_data(show_spinner=False)
def _target_samples(n=4000, seed=0):
    """A 1-D two-bump distribution: small enough to see, hard enough to be interesting
    (a single Gaussian would let every method look equally good)."""
    rng = np.random.default_rng(seed)
    pick = rng.random(n) < 0.35
    return np.where(pick, rng.normal(-1.6, 0.35, n), rng.normal(1.1, 0.55, n))


tab_live, tab_theory, tab_cmp, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Live", "📘 Theory", "⚖️ Compare", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

# --------------------------------------------------------------------------- #
with tab_live:
    st.markdown("#### Two routes to the same distribution")
    st.markdown(
        "The grey histogram is the **target** — a distribution with two bumps that no "
        "single Gaussian can fake. Below, two families try to reproduce it from opposite "
        "directions. Neither is a neural network here; both are the *mechanism*, stripped "
        "to arithmetic you can read."
    )

    c = st.columns(3)
    bins = c[0].slider("Autoregressive: bins (resolution of the discretisation)",
                       8, 120, 40, 4, key="gen_bins",
                       help="An autoregressive model over a continuous variable has to "
                            "chop it into a finite vocabulary first — exactly what a "
                            "tokenizer does to text.")
    steps = c[1].slider("Diffusion: denoising steps $T$", 2, 200, 60, 2, key="gen_steps",
                        help="How many small reversals the model takes to walk pure noise "
                             "back to data. More steps = better samples, slower sampling.")
    nsamp = c[2].slider("Samples drawn", 200, 5000, 2000, 100, key="gen_n")

    data = _target_samples()
    lo, hi = -3.2, 3.2
    rng = np.random.default_rng(7)

    # --- autoregressive: discretise, learn the histogram, sample from it ------ #
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(data, bins=edges)
    probs = counts / counts.sum()
    idx = rng.choice(bins, size=nsamp, p=probs)
    ar = edges[idx] + rng.random(nsamp) * (edges[1] - edges[0])

    # --- diffusion: forward noising has a closed form, so learn the reversal --- #
    # Variance-preserving schedule: x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.
    # The exact posterior mean is available for this toy target, so the "model" here is
    # the true score - which is the point: it isolates the *sampler* from the network.
    betas = np.linspace(1e-4, 0.02, steps)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)

    def _score(x, t):
        """d/dx log q(x_t) for a two-Gaussian mixture pushed through the forward process."""
        s, w, mu, sd = np.sqrt(abar[t]), np.array([0.35, 0.65]), \
            np.array([-1.6, 1.1]), np.array([0.35, 0.55])
        m, v = s * mu, abar[t] * sd ** 2 + (1 - abar[t])
        d = x[:, None] - m[None, :]
        comp = w / np.sqrt(2 * np.pi * v) * np.exp(-0.5 * d ** 2 / v)
        post = comp / np.clip(comp.sum(1, keepdims=True), 1e-30, None)
        return -(post * d / v).sum(1)

    x = rng.normal(0, 1, nsamp)                      # start from pure noise
    for t in range(steps - 1, -1, -1):
        mean = (x + betas[t] * _score(x, t)) / np.sqrt(alphas[t])
        x = mean + (np.sqrt(betas[t]) * rng.normal(0, 1, nsamp) if t else 0.0)
    dif = x

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    for a, samp, name, col in ((ax[0], ar, "Autoregressive (binned)", "#2F7BEA"),
                               (ax[1], dif, "Diffusion (denoised)", "#9A6A2A")):
        a.hist(data, bins=70, range=(lo, hi), density=True, color="#C9CCD1",
               label="target", alpha=.85)
        a.hist(samp, bins=70, range=(lo, hi), density=True, histtype="step",
               color=col, lw=2, label=name)
        a.set_title(name, fontsize=10)
        a.legend(fontsize=8, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    m = st.columns(3)
    m[0].metric("AR vocabulary size", f"{bins}",
                help="Its resolution can never beat one bin — visible as steps in the outline.")
    m[1].metric("Diffusion network calls / sample", f"{steps}",
                help="One forward pass per step. This is why diffusion sampling is slow.")
    m[2].metric("AR network calls / sample", "1",
                help="One draw for a single-variable target. For a length-T sequence it "
                     "would be T — the same trade, mirrored.")

    st.info(
        "**Turn the bins down to 8.** The autoregressive outline goes visibly blocky — it "
        "cannot represent anything finer than one bin, because it made the variable "
        "*discrete* to get an exact likelihood. That is the same trade a tokenizer makes.\n\n"
        "**Now set diffusion steps to 2.** The samples collapse toward a single blob: each "
        "reversal step is only valid when it is *small*, so too few steps and the walk back "
        "from noise cannot resolve two separate bumps. Raise it to 200 and the bumps sharpen.",
        icon=":material/lightbulb:")

# --------------------------------------------------------------------------- #
with tab_theory:
    st.markdown(_MAP_SVG, unsafe_allow_html=True)
    st.markdown(r"""
## 1. The one problem all four are solving

Learn a distribution $p_\theta(x)$ close to the data distribution $p_{\text{data}}(x)$, and
be able to **sample** from it. The honest way to fit it is maximum likelihood:

$$ \max_\theta \; \mathbb E_{x\sim p_{\text{data}}}\big[\log p_\theta(x)\big] $$

The difficulty is that for anything interesting, $p_\theta(x)$ cannot be written down. A
probability must integrate to 1, so any expression needs a normalising constant

$$ Z_\theta = \int \exp\big(f_\theta(x)\big)\,dx $$

and that integral runs over *every possible image or sentence*. It is hopeless to compute.
**All four families are strategies for avoiding $Z_\theta$** — and each pays a different
price for the escape. That is the whole story; the rest is detail.

## 2. Autoregressive — factorise, and the constant vanishes

Split $x$ into an ordered sequence and apply the chain rule, which is an exact identity:

$$ p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}) $$

Each factor is a distribution over **one** variable — for text, a softmax over ~50k tokens.
Normalising *that* is a sum over 50k numbers, which is trivial. The intractable integral
became a cheap sum, and the likelihood is **exact**, not a bound.

**Price:** you must impose an order, the variable must be discrete (or you must bin it, as
the Live tab does), and generation is inherently sequential — $T$ forward passes for $T$
tokens, which is exactly the KV-cache problem from the Tiny GPT page.

**This is what you built.** Nothing else in this lab is generative.

## 3. VAE — bound the thing you cannot compute

Introduce a latent $z$ (a short code, say 128 numbers) and write

$$ p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz $$

still intractable. So train an **encoder** $q_\phi(z\mid x)$ to guess which $z$ produced
$x$, and optimise a *lower bound* on the likelihood — the ELBO:

$$ \log p_\theta(x)\;\ge\;\underbrace{\mathbb E_{q_\phi}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruct it}}\;-\;\underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{keep the code tidy}} $$

The KL term is the one from the Information theory page (X5): it pulls the encoder's output
toward a plain $\mathcal N(0,I)$, so afterwards you can **sample $z$ from a Gaussian** and
decode it. Without that term you would have an autoencoder — good at compression, unable to
generate, because you would not know which codes are valid.

**Price:** a bound is not the thing. The gap between the ELBO and $\log p_\theta(x)$ is
itself a KL divergence, and squeezing the code through a Gaussian bottleneck averages away
detail — the classic reason VAE images look **blurry**.

## 4. GAN — stop computing likelihood entirely

Abandon $p_\theta(x)$ as a number. Keep only a sampler: a generator $G$ that maps noise to
data. Judge it with a second network, a discriminator $D$, trained to tell real from fake:

$$ \min_G \max_D \; \mathbb E_{x\sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb E_{z}\big[\log\big(1 - D(G(z))\big)\big] $$

At the optimum $D$ this objective equals the **Jensen–Shannon divergence** between real and
generated distributions — so the generator is minimising a genuine distance, without ever
evaluating a density. Samples are drawn in **one** forward pass, and they are sharp: nothing
is averaged, so there is no blur to produce.

**Price:** it is a two-player game, not a minimisation, and games need not converge. Both
networks chase a moving target; the notorious failure is **mode collapse** — the generator
finds one output that fools $D$ and emits only that. On the Live tab's target, a collapsed
GAN would produce one bump and ignore the other. There is also no likelihood, so you cannot
even score how well it did.

## 5. Diffusion — make the hard step small enough to be easy

The insight: learning to generate an image from nothing is hard, but learning to make a
slightly noisy image *slightly* cleaner is easy. So define a **forward process** that
destroys data over $T$ steps, adding a little Gaussian noise each time:

$$ q(x_t \mid x_{t-1}) = \mathcal N\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big) $$

This has no parameters — it is just noising — and it has a closed form that jumps straight
to any step, which is what makes training cheap:

$$ x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{s\le t}(1-\beta_s) $$

Then learn the **reverse**. Because each forward step is tiny, its reversal is also
approximately Gaussian, and the whole job reduces to predicting the noise that was added:

$$ \mathcal L = \mathbb E_{t,x_0,\varepsilon}\Big[\big\|\varepsilon - \varepsilon_\theta(x_t,t)\big\|^2\Big] $$

A plain squared error — the loss from the very first page of this lab. That is the trick
that made diffusion take over: **a stable regression objective**, no adversarial game, no
posterior collapse, and it still corresponds to a bound on $\log p_\theta(x)$.

**Price:** sampling calls the network once per step. Early models needed 1000; distillation
and better solvers have cut that to 1–8, which is why image generation stopped feeling slow.

## 6. Why text and images went different ways

Not fashion — the data forced it.

**Text is discrete and already ordered.** Left-to-right is how language is produced and
read, so the autoregressive factorisation costs nothing and buys an exact likelihood. And
diffusion's whole mechanism — add a *little* noise — has no clean meaning on a token: there
is no "slightly the word cat".

**Images are continuous with no natural order.** Generating pixel 1, then pixel 2, is an
arbitrary raster scan that throws away 2-D structure and needs a million sequential steps.
Diffusion instead refines the **whole canvas at once**, coarse structure first, detail last,
which matches how an image is actually organised.

The two are converging at the edges — diffusion language models and autoregressive image
models both exist — but the defaults are set by this, not by history.

## 7. Where the modern systems actually sit

- **GPT, Claude, Llama** — autoregressive transformers. Exactly section 2.
- **Stable Diffusion, Imagen, DALL·E 3, Midjourney** — diffusion, usually in the *latent*
  space of a VAE rather than on raw pixels, which is the two families combined: the VAE
  compresses 512×512×3 down to a small grid, and diffusion runs there instead. That is
  what the "latent" in Latent Diffusion means.
- **GANs** — largely displaced for open-ended generation, still used where one-pass speed
  matters: super-resolution, face editing, and as the *decoder* inside some audio codecs.
- **Flow matching / rectified flow** — the current direction. Same destroy-and-reverse idea
  as diffusion, but learning a straight-line velocity field instead of a noise-removal
  step, so the path from noise to data is shorter and needs fewer calls.
""")

# --------------------------------------------------------------------------- #
with tab_cmp:
    st.markdown("#### The four, side by side")
    st.markdown("""
| | **Autoregressive** | **VAE** | **GAN** | **Diffusion** |
|---|---|---|---|---|
| **How it dodges $Z_\\theta$** | factorises it away exactly | bounds it (ELBO) | never computes it | bounds it, in tiny steps |
| **Objective** | exact log-likelihood | lower bound | adversarial (JS divergence) | noise-prediction MSE |
| **Training stability** | very stable | stable | **unstable** — a two-player game | very stable |
| **Sampling cost** | $T$ passes (one per token) | **1** pass | **1** pass | $T$ passes (one per step) |
| **Sample quality** | excellent for text | blurry | sharp but can collapse | state of the art |
| **Can score a sample?** | **yes**, exactly | bound only | **no** | bound only |
| **Natural data type** | discrete sequences | continuous | continuous | continuous |
| **Signature failure** | drifts / repeats | blur | mode collapse | slow sampling |
| **Used today for** | text, code, audio tokens | latent spaces *inside* diffusion | fast one-pass niches | images, video, audio |
""")
    st.info(
        "**Read the table down the 'sampling cost' row.** Autoregression and diffusion pay "
        "the same price — many sequential network calls — for opposite reasons: one walks "
        "along the *sequence*, the other along the *noise schedule*. VAEs and GANs are the "
        "cheap ones, and both gave something up for it (a bound, and any likelihood at all).",
        icon=":material/insights:")

    st.markdown("#### One sentence each, if you remember nothing else")
    st.markdown(
        "- **Autoregressive** — predict the next piece, forever. Exact, ordered, sequential.\n"
        "- **VAE** — squeeze through a tidy Gaussian code so you can sample the code.\n"
        "- **GAN** — two networks argue; the forger gets good.\n"
        "- **Diffusion** — learn to undo noise, one easy step at a time."
    )

# --------------------------------------------------------------------------- #
with tab_quiz:
    lessons.render_quiz([
        lessons.Question(
            "Why can an autoregressive model compute an *exact* likelihood when the others cannot?",
            ["It uses a bigger network",
             "The chain rule turns one intractable normalising integral into T cheap softmax sums",
             "It ignores the normalising constant",
             "Because it is trained with cross-entropy"],
            1,
            "Each factor is a distribution over one variable, so normalising means summing "
            "over the vocabulary — not integrating over all possible sequences."),
        lessons.Question(
            "A VAE is trained without the KL term. What breaks?",
            ["Nothing — the KL term is a regulariser",
             "Reconstructions get worse",
             "It becomes an autoencoder: it can compress and rebuild, but you no longer know "
             "which latents are valid, so you cannot sample",
             "Training diverges"],
            2,
            "The KL term is what makes the latent space match the prior you later sample from. "
            "Without it, reconstruction is fine and *generation* is impossible."),
        lessons.Question(
            "Mode collapse is a failure of which family, and what does it look like?",
            ["Diffusion — samples stay noisy",
             "GAN — the generator emits one convincing output and abandons the rest of the distribution",
             "VAE — samples become blurry",
             "Autoregressive — the text repeats"],
            1,
            "A GAN only has to fool the discriminator. Covering the whole distribution is not "
            "in its objective, so one good fake can be a local optimum."),
        lessons.Question(
            "Why is diffusion's training objective easier than a GAN's?",
            ["It uses fewer parameters",
             "It is a plain regression — predict the added noise — so there is a fixed target, "
             "not an opponent that moves",
             "It needs no data",
             "It trains for fewer epochs"],
            1,
            "Squared error against a known epsilon. A GAN's target moves because the "
            "discriminator is learning at the same time."),
        lessons.Question(
            "Why is diffusion a poor fit for text, in one word?",
            ["Speed", "Discreteness", "Memory", "Ordering"],
            1,
            "Diffusion needs 'slightly noisier' to mean something. Tokens are discrete — there "
            "is no small perturbation of the word 'cat'."),
        lessons.Question(
            "In Stable Diffusion, what is the VAE doing?",
            ["Generating the image",
             "Compressing the image to a small latent grid so diffusion runs there instead of on pixels",
             "Scoring the samples",
             "Replacing the text encoder"],
            1,
            "That is what 'latent' in Latent Diffusion means — two families combined, each "
            "doing the part it is good at."),
    ], prefix="gen")

# --------------------------------------------------------------------------- #
with tab_tasks:
    st.markdown("#### Tasks")
    st.markdown(
        "1. On the **Live** tab set bins to 8, then 120. Explain the blockiness in terms of "
        "what an autoregressive model must do to a continuous variable.\n"
        "2. Set diffusion steps to 2, then 200. Why does a two-step reversal fail to "
        "separate the bumps, when the same score function works at 200?\n"
        "3. Both panels reach the same target from opposite ends. State, in one sentence "
        "each, what the *sequential* cost is buying in each case.\n"
        "4. Your Tiny GPT model assigns a probability to any string. Write the expression "
        "for the probability of `the cat` and say why no GAN could give you that number.\n"
        "5. Suppose you must generate 512×512 images and you need a likelihood score for "
        "each sample (say, for outlier detection). Which family, and what do you give up?"
    )
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** An autoregressive model needs a *finite* set of outcomes per step so its
softmax can normalise. A continuous variable must therefore be discretised, and the model
cannot represent structure finer than one bin — with 8 bins the two bumps become eight flat
plateaus. This is precisely what tokenization does to text: it is the same compromise,
made invisible by the fact that text was already discrete.

**2.** The reversal is only approximately Gaussian when the forward step is *small*. The
derivation assumes $\beta_t \to 0$; with $T=2$ each $\beta_t$ is huge, the true reverse
distribution is strongly multi-modal, and a Gaussian step cannot split into two bumps — so
the samples pile up between them. The score function is the same at $T=200$; what changes
is that each step is now small enough for the Gaussian assumption to hold.

**3.** Autoregressive: the $T$ passes buy an **exact likelihood and a valid ordering** —
each pass conditions on everything already generated. Diffusion: the $T$ passes buy
**stability** — each one only has to make a slightly noisy sample slightly cleaner, which
is an easy regression instead of a hard one-shot generation.""",
        label="Live 1–3")
    lessons.solution(
        r"""**4.** $p(\texttt{the cat}) = \prod_{t} p(x_t \mid x_{<t})$ — multiply the eight
per-character probabilities the model already outputs. A GAN has no such quantity: it is
only a sampler $x = G(z)$, and it is never trained to evaluate a density. Asking a GAN
"how probable is this string" has no answer even in principle — which is also why GANs
cannot be compared by perplexity.

**5.** **Autoregressive** is the only family that gives an exact likelihood, so if the score
must be real you take it — and you give up a great deal of image quality and speed, because
generating 262,144 pixels sequentially is both slow and a poor fit for 2-D data. The
realistic answer is diffusion for quality plus an **ELBO** as an approximate score,
accepting that a bound can rank samples but cannot be trusted as an absolute number.""",
        label="Concept 4–5")

# --------------------------------------------------------------------------- #
with tab_ref:
    st.subheader("Reading & references")
    st.markdown("""
**Primary papers**

- Kingma & Welling (2013), *Auto-Encoding Variational Bayes* — the VAE and the ELBO.
- Goodfellow et al. (2014), *Generative Adversarial Networks* — the minimax game.
- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models* — the paper that
  reduced diffusion training to noise-prediction MSE and made it work.
- Song et al. (2021), *Score-Based Generative Modeling through SDEs* — unifies diffusion
  and score matching; the continuous-time view.
- Rombach et al. (2022), *High-Resolution Image Synthesis with Latent Diffusion Models* —
  Stable Diffusion; diffusion inside a VAE's latent space.
- Lipman et al. (2023), *Flow Matching for Generative Modeling* — the current direction.

**Where this lab already covers the machinery**

- Chain rule and conditional probability — **X3 · Probability**
- KL divergence and cross-entropy (the ELBO's second term) — **X5 · Information theory**
- Squared error and gradient descent (the diffusion objective) — **Level 1 · The neuron**
- The autoregressive model itself — **Level 5 · Tiny GPT**, §9

**Caveat.** This page is a map, not a manual. Every family here has a literature of its own,
and the Live tab uses the *exact* score of a toy distribution rather than a learned network
— it demonstrates the sampler, not the training.
""")
