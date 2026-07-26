import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import lessons
import math_lessons2

LESSON = math_lessons2.PROBABILITY

DISTS = {
    "Uniform [0,1]": lambda rng, shape: rng.uniform(0, 1, shape),
    "Exponential": lambda rng, shape: rng.exponential(1.0, shape),
    "Bernoulli(0.3)": lambda rng, shape: (rng.random(shape) < 0.3).astype(float),
}

USES = {
    "Bernoulli": "Any single yes/no outcome — spam/not, click/no-click, fraud/legit. The output of logistic regression and a sigmoid neuron.",
    "Binomial": "Counts of successes over n independent trials — conversions out of n visitors (A/B tests), defects per batch.",
    "Poisson": "Counts of rare, independent events per interval — arrivals per minute, photons on a sensor, typos per page. Poisson regression models counts.",
    "Geometric": "How many trials until the first success — attempts until a user converts or a packet is delivered (memoryless).",
    "Gaussian": "Noise, regression residuals, standardized features, weight init, the CLT limit, VAE latents — the default when unsure.",
    "Beta": "An unknown probability/rate in [0,1] — click/conversion rates; the prior in Bayesian A/B testing & Thompson-sampling bandits.",
    "Gamma": "Positive, skewed amounts — time-to-failure, claim sizes, rainfall, the wait for k events; conjugate prior for a Poisson rate.",
}


KIND = {
    "Bernoulli": "d", "Binomial": "d", "Poisson": "d", "Geometric": "d",
    "Categorical": "d", "Uniform (discrete)": "d",
    "Gaussian": "c", "Uniform (continuous)": "c", "Exponential": "c",
    "Laplace": "c", "Beta": "c", "Gamma": "c",
}

TEX = {
    "Bernoulli": r"P(X=k)=p^{k}(1-p)^{1-k},\;k\in\{0,1\}",
    "Binomial": r"P(X=k)=\binom{n}{k}p^{k}(1-p)^{n-k}",
    "Poisson": r"P(X=k)=\frac{\lambda^{k}e^{-\lambda}}{k!}",
    "Geometric": r"P(X=k)=(1-p)^{k-1}p,\;k=1,2,\dots",
    "Categorical": r"P(X=i)=p_i,\quad \textstyle\sum_i p_i=1",
    "Uniform (discrete)": r"P(X=k)=\frac{1}{b-a+1},\;k=a,\dots,b",
    "Gaussian": r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    "Uniform (continuous)": r"f(x)=\frac{1}{b-a},\;a\le x\le b",
    "Exponential": r"f(x)=\lambda e^{-\lambda x},\;x\ge 0",
    "Laplace": r"f(x)=\frac{1}{2b}e^{-|x-\mu|/b}",
    "Beta": r"f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\;0<x<1",
    "Gamma": r"f(x)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},\;x>0",
}

USES = {
    "Bernoulli": "Any single yes/no outcome — spam/not, click/no-click, fraud/legit. The output of logistic regression and a sigmoid neuron.",
    "Binomial": "Counts of successes over n independent trials — conversions out of n visitors (A/B tests), defects per batch.",
    "Poisson": "Counts of rare, independent events per interval — arrivals per minute, photons on a sensor, typos per page. Poisson regression models counts.",
    "Geometric": "How many trials until the first success — attempts until a user converts or a packet is delivered (memoryless).",
    "Categorical": "One draw from k labelled outcomes — **exactly what a softmax layer outputs**: next-token over a vocabulary, class over k classes. Cross-entropy is its negative log-likelihood.",
    "Uniform (discrete)": "Every outcome equally likely — a fair die, a random index, shuffling, random seeds. The maximum-entropy choice when you know nothing.",
    "Gaussian": "Noise, regression residuals, standardized features, weight init, the CLT limit, VAE latents — the default when unsure.",
    "Uniform (continuous)": "A value known only to lie in a range — random init bounds, dropout/sampling draws, the base for inverse-transform sampling.",
    "Exponential": "Waiting time until the next event of a Poisson process — time between arrivals, time-to-failure. **Memoryless**: waiting longer tells you nothing.",
    "Laplace": "A sharply peaked, heavy-tailed alternative to the Gaussian. Its negative log-likelihood is the **absolute error (L1)**, and as a prior it gives **lasso** — which is why L1 produces sparsity.",
    "Beta": "An unknown probability/rate in [0,1] — click/conversion rates; the prior in Bayesian A/B testing & Thompson-sampling bandits.",
    "Gamma": "Positive, skewed amounts — time-to-failure, claim sizes, rainfall, the wait for k events; conjugate prior for a Poisson rate.",
}


def _explorer():
    kind = st.radio("family", ["Discrete (counts / categories)", "Continuous (measurements)"],
                    horizontal=True, key="ex_kind")
    want = "d" if kind.startswith("Discrete") else "c"
    options = [k for k, v in KIND.items() if v == want]

    left, right = st.columns([0.42, 0.58])
    probs = None
    with left:
        dist = st.selectbox("distribution", options, key=f"ex_dist_{want}")
        if dist == "Bernoulli":
            p = st.slider("p", 0.0, 1.0, 0.5, 0.05, key="ex_bern_p")
            xs, y, mean, var, disc = np.array([0, 1]), np.array([1 - p, p]), p, p * (1 - p), True
        elif dist == "Binomial":
            n = st.slider("n", 1, 40, 10, key="ex_bin_n")
            p = st.slider("p", 0.0, 1.0, 0.5, 0.05, key="ex_bin_p")
            xs = np.arange(0, n + 1)
            y = np.array([math.comb(n, int(k)) * p ** k * (1 - p) ** (n - k) for k in xs])
            mean, var, disc = n * p, n * p * (1 - p), True
        elif dist == "Poisson":
            lam = st.slider("λ (rate)", 0.5, 30.0, 4.0, 0.5, key="ex_pois_lam")
            xs = np.arange(0, int(lam + 4 * np.sqrt(lam) + 5))
            y = np.exp(-lam + xs * np.log(lam) - np.array([math.lgamma(int(k) + 1) for k in xs]))
            mean, var, disc = lam, lam, True
        elif dist == "Geometric":
            p = st.slider("p", 0.05, 1.0, 0.3, 0.05, key="ex_geo_p")
            xs = np.arange(1, 31)
            y, mean, var, disc = (1 - p) ** (xs - 1) * p, 1 / p, (1 - p) / p ** 2, True
        elif dist == "Categorical":
            k = st.slider("categories k", 2, 6, 4, key="ex_cat_k")
            st.caption("relative weights (auto-normalized to sum to 1)")
            cols = st.columns(k)
            w = np.array([cols[i].number_input(f"w{i+1}", 0.0, 10.0,
                                               float([3.0, 1.0, 2.0, 1.5, 1.0, 0.5][i]), 0.5,
                                               key=f"ex_cat_w{i}") for i in range(k)])
            w = np.maximum(w, 1e-9); probs = w / w.sum()
            xs, y = np.arange(1, k + 1), probs
            mean = float((xs * probs).sum())
            var = float((probs * (xs - mean) ** 2).sum())
            disc = True
        else:  # Uniform (discrete)
            a = st.slider("a (low)", 0, 10, 1, key="ex_ud_a")
            b = st.slider("b (high)", 1, 20, 6, key="ex_ud_b")
            b = max(b, a)
            xs = np.arange(a, b + 1)
            y = np.full(xs.shape, 1.0 / len(xs))
            mean, var, disc = (a + b) / 2, ((b - a + 1) ** 2 - 1) / 12, True

        if want == "c":
            if dist == "Gaussian":
                mu = st.slider("μ (mean)", -5.0, 5.0, 0.0, 0.5, key="ex_g_mu")
                sigma = st.slider("σ (std)", 0.3, 4.0, 1.0, 0.1, key="ex_g_sig")
                xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
                y = np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
                mean, var, disc = mu, sigma ** 2, False
            elif dist == "Uniform (continuous)":
                a = st.slider("a (low)", -5.0, 5.0, 0.0, 0.5, key="ex_uc_a")
                b = st.slider("b (high)", -5.0, 10.0, 1.0, 0.5, key="ex_uc_b")
                b = max(b, a + 1e-6)
                xs = np.linspace(a - 0.25 * (b - a) - 0.2, b + 0.25 * (b - a) + 0.2, 400)
                y = np.where((xs >= a) & (xs <= b), 1.0 / (b - a), 0.0)
                mean, var, disc = (a + b) / 2, (b - a) ** 2 / 12, False
            elif dist == "Exponential":
                lam = st.slider("λ (rate)", 0.2, 5.0, 1.0, 0.1, key="ex_exp_lam")
                xs = np.linspace(0, 6 / lam, 300)
                y, mean, var, disc = lam * np.exp(-lam * xs), 1 / lam, 1 / lam ** 2, False
            elif dist == "Laplace":
                mu = st.slider("μ (location)", -5.0, 5.0, 0.0, 0.5, key="ex_lap_mu")
                bsc = st.slider("b (scale)", 0.2, 3.0, 1.0, 0.1, key="ex_lap_b")
                xs = np.linspace(mu - 8 * bsc, mu + 8 * bsc, 400)
                y = np.exp(-np.abs(xs - mu) / bsc) / (2 * bsc)
                mean, var, disc = mu, 2 * bsc ** 2, False
            elif dist == "Beta":
                a = st.slider("α", 0.5, 6.0, 2.0, 0.5, key="ex_beta_a")
                b = st.slider("β", 0.5, 6.0, 2.0, 0.5, key="ex_beta_b")
                xs = np.linspace(0.001, 0.999, 300)
                B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
                y = xs ** (a - 1) * (1 - xs) ** (b - 1) / B
                mean, var, disc = a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1)), False
            else:  # Gamma
                a = st.slider("shape α", 0.5, 8.0, 2.0, 0.5, key="ex_gam_a")
                rate = st.slider("rate β", 0.3, 4.0, 1.0, 0.1, key="ex_gam_b")
                xs = np.linspace(0.001, a / rate + 4 * np.sqrt(a) / rate + 1, 300)
                y = np.exp(a * np.log(rate) + (a - 1) * np.log(xs) - rate * xs - math.lgamma(a))
                mean, var, disc = a / rate, a / rate ** 2, False

    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        if disc:
            ax.bar(xs, y, color="#185FA5",
                   width=0.4 if dist in ("Bernoulli",) else 0.85)
            ax.set_ylabel("P(X = k)")
            if len(xs) <= 12:
                ax.set_xticks(xs)
        else:
            ax.plot(xs, y, color="#185FA5", lw=2)
            ax.fill_between(xs, y, alpha=0.25, color="#185FA5")
            ax.set_ylabel("density")
        ax.axvline(mean, color="#C0507A", ls="--", lw=1.5, label=f"mean = {mean:.2f}")
        sd = float(np.sqrt(max(var, 0.0)))
        ax.axvspan(mean - sd, mean + sd, color="#C0507A", alpha=.08, label="±1 std")
        ax.set_title(f"{dist} distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    st.latex(TEX[dist])
    c1, c2, c3 = st.columns(3)
    c1.metric("mean", f"{mean:.3f}")
    c2.metric("variance", f"{var:.3f}")
    c3.metric("std", f"{np.sqrt(max(var,0.0)):.3f}")
    if probs is not None:
        st.caption("normalized probabilities: " + ",  ".join(f"p{i+1}={v:.3f}" for i, v in enumerate(probs))
                   + f"   (sum = {probs.sum():.2f})")
        st.caption("*Mean/variance treat the categories as the numbers 1…k; for truly nominal "
                   "labels only the probabilities themselves are meaningful.*")
    st.info(USES[dist], icon=":material/lightbulb:")


def _clt():
    left, right = st.columns([0.42, 0.58])
    with left:
        name = st.selectbox("base distribution (not Gaussian!)", list(DISTS), key="clt_base")
        n = st.slider("sample size n (averaged)", 1, 50, 5, key="p_n")
        seed = st.number_input("seed", 0, 9999, 0, key="p_seed")
        trials = 4000
    rng = np.random.default_rng(int(seed))
    samples = DISTS[name](rng, (trials, int(n)))
    means = samples.mean(axis=1)
    m, s = float(means.mean()), float(means.std())
    with right:
        fig, ax = plt.subplots(figsize=(4.7, 4.0))
        ax.hist(means, bins=40, density=True, color="#185FA5", alpha=0.75, label=f"mean of n={n}")
        if s > 1e-9:
            gx = np.linspace(means.min(), means.max(), 200)
            ax.plot(gx, np.exp(-(gx - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)),
                    color="#C0507A", lw=2, label="Gaussian fit")
        ax.set_title("distribution of the sample mean")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("n (averaged)", int(n))
    c2.metric("mean of means", f"{m:.3f}")
    c3.metric("std of means", f"{s:.3f}")
    st.info("At **n = 1** you see the base distribution's true (often skewed) shape. As **n "
            "grows**, the average becomes **Gaussian** (CLT) and its spread shrinks ~1/√n.",
            icon=":material/lightbulb:")


st.title("X3 · Probability — playground & lesson")
st.caption("Explore the key distributions (shape, mean, variance, where they're used), or "
           "watch the Central Limit Theorem turn any distribution's average into a Gaussian.")

lessons.predict(
    'Average **n** samples from a *skewed* distribution and increase n. What shape does the distribution of the average approach — and what happens to its spread?',
    "It approaches a **Gaussian** (the Central Limit Theorem) no matter the base shape, and its spread shrinks like 1/√n. That's why sample means are so well-behaved — and a big reason the Gaussian shows up everywhere.",
)

tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)


def _bayes():
    st.markdown("**Bayes' rule updates a belief with evidence.** The classic shock: a test that "
                "is 99 % accurate can still be *usually wrong* when the disease is rare — because "
                "the few true positives are swamped by false positives from the huge healthy "
                "majority.")
    st.latex(r"P(D\mid +)=\frac{P(+\mid D)\,P(D)}{P(+\mid D)P(D)+P(+\mid \neg D)P(\neg D)}")
    c = st.columns(3)
    prev = c[0].slider("base rate P(D) — how common it is", 0.001, 0.50, 0.010, 0.001,
                       format="%.3f", key="b_prev")
    sens = c[1].slider("sensitivity P(+|D) — catches the sick", 0.50, 1.0, 0.99, 0.01, key="b_sens")
    spec = c[2].slider("specificity P(−|¬D) — clears the healthy", 0.50, 1.0, 0.99, 0.01, key="b_spec")

    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    post = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fn = (1 - sens) * prev
    tn = spec * (1 - prev)
    post_neg = fn / (fn + tn) if (fn + tn) > 0 else 0.0

    m = st.columns(3)
    m[0].metric("P(sick | POSITIVE)", f"{post*100:.1f}%")
    m[1].metric("P(sick | negative)", f"{post_neg*100:.2f}%")
    m[2].metric("prior → posterior", f"{prev*100:.1f}% → {post*100:.1f}%",
                delta=f"×{post/prev:.1f}" if prev > 0 else None)
    st.latex(rf"P(D\mid+)=\frac{{{sens:.2f}\times{prev:.3f}}}"
             rf"{{{sens:.2f}\times{prev:.3f}+{1-spec:.2f}\times{1-prev:.3f}}}"
             rf"=\frac{{{tp:.5f}}}{{{tp+fp:.5f}}}={post:.4f}")

    N = 10000
    counts = dict(TP=tp*N, FP=fp*N, FN=fn*N, TN=tn*N)
    st.markdown(
        f"**Natural frequencies — imagine 10 000 people:**\n\n"
        f"| | tests **+** | tests **−** | total |\n|---|---|---|---|\n"
        f"| **has it** | {counts['TP']:.0f} ✅ | {counts['FN']:.0f} ❌ missed | {prev*N:.0f} |\n"
        f"| **healthy** | {counts['FP']:.0f} ❌ false alarm | {counts['TN']:.0f} ✅ | {(1-prev)*N:.0f} |\n"
        f"| **total** | {(tp+fp)*N:.0f} | {(fn+tn)*N:.0f} | {N} |\n\n"
        f"Of the **{(tp+fp)*N:.0f}** people who test positive, only **{counts['TP']:.0f}** are "
        f"actually sick → **{post*100:.1f} %**."
    )
    fig, ax = plt.subplots(figsize=(6.8, 1.5))
    left = 0.0
    for lbl, val, col in [("true +", tp, "#1D9E75"), ("false +", fp, "#C0507A")]:
        w = val / (tp + fp) * 100 if (tp + fp) > 0 else 0
        ax.barh([0], [w], left=[left], color=col, height=.6)
        if w > 6:
            ax.text(left + w / 2, 0, f"{lbl}\n{w:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += w
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("of everyone who tests POSITIVE (%)", fontsize=9)
    ax.tick_params(labelsize=8); fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.info("**Drag the base rate to 0.001** and even a 99 %-accurate test gives only ~9 % — the "
            "*base-rate fallacy*. Evidence multiplies a prior; it does not replace it. This is why "
            "screening rare conditions needs very high **specificity**, and it is the same maths "
            "behind naive Bayes classifiers and Bayesian priors as regularization (§13).",
            icon=":material/psychology:")


def _mle():
    st.markdown("**Maximum likelihood is where loss functions come from.** Pick the parameter "
                "that makes the observed data *most probable*. Below: flip a coin $n$ times, see "
                "$k$ heads, and ask which $p$ best explains it.")
    c = st.columns(3)
    n = c[0].slider("flips n", 5, 200, 20, 5, key="m_n")
    k = c[1].slider("heads observed k", 0, 200, 13, 1, key="m_k")
    k = min(k, n)
    show_log = c[2].toggle("show log-likelihood", value=True, key="m_log")

    ps = np.linspace(1e-4, 1 - 1e-4, 500)
    loglik = k * np.log(ps) + (n - k) * np.log(1 - ps)
    lik = np.exp(loglik - loglik.max())
    mle = k / n

    m = st.columns(3)
    m[0].metric("MLE  p̂ = k/n", f"{mle:.4f}")
    m[1].metric("log-likelihood at p̂", f"{k*np.log(max(mle,1e-12)) + (n-k)*np.log(max(1-mle,1e-12)):.3f}")
    m[2].metric("negative log-lik (the loss)",
                f"{-(k*np.log(max(mle,1e-12)) + (n-k)*np.log(max(1-mle,1e-12))):.3f}")
    st.latex(rf"\mathcal L(p)=p^{{{k}}}(1-p)^{{{n-k}}}\quad\Rightarrow\quad"
             rf"\log\mathcal L = {k}\log p + {n-k}\log(1-p)")
    st.latex(rf"\frac{{d\log\mathcal L}}{{dp}}=\frac{{{k}}}{{p}}-\frac{{{n-k}}}{{1-p}}=0"
             rf"\;\Longrightarrow\; \hat p=\frac{{k}}{{n}}=\frac{{{k}}}{{{n}}}={mle:.4f}")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(ps, loglik if show_log else lik, lw=2.2, color="#185FA5")
    ax.axvline(mle, color="#1D9E75", lw=2, ls="--", label=f"MLE p̂ = {mle:.3f}")
    ax.plot([mle], [(k*np.log(mle) + (n-k)*np.log(1-mle)) if show_log else 1.0], "o",
            ms=9, color="#111827", zorder=5)
    ax.set_xlabel("candidate p", fontsize=9)
    ax.set_ylabel("log-likelihood" if show_log else "likelihood (scaled)", fontsize=9)
    ax.set_title("the parameter that best explains the data", fontsize=9.5)
    if show_log:
        ax.set_ylim(max(loglik.max() - 6 * max(1.0, n / 20), loglik.min()), loglik.max() + .5)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8); fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.info("**The peak is the estimate — and its negative is the loss you already know.** "
            "Minimizing $-\\log\\mathcal L$ for a Bernoulli *is* **binary cross-entropy** (ML "
            "**M2**, Math **X5**); doing the same for Gaussian noise gives **squared error** "
            "(M1). Note the curve gets **sharper as n grows** — more data means less uncertainty "
            "about $p$. That width is the standard error, $\\sqrt{p(1-p)/n}$.",
            icon=":material/insights:")



def _init_variance():
    """Variance algebra (§10) applied to a deep net: where sqrt(1/n) and sqrt(2/n) come
    from, and what happens when you guess instead. Same experiment as e10, live."""
    st.markdown("**Two variance rules, one consequence.** A unit computes "
                "$z=\\sum_i w_i x_i$ over `fan_in` inputs, so §10 gives "
                "$\\operatorname{Var}(z)=n\\,\\operatorname{Var}(w)\\,"
                "\\operatorname{Var}(x)$. Demand that a layer neither grows nor shrinks "
                "the signal and the initialisation scale is forced — there is nothing to "
                "tune. Watch what a wrong choice does with depth.")

    c = st.columns(4)
    depth = c[0].slider("layers", 2, 30, 12, 1, key="iv_depth")
    width = c[1].slider("units per layer (fan_in)", 16, 512, 256, 16, key="iv_width")
    act_name = c[2].selectbox("nonlinearity", ["ReLU", "tanh", "none (linear)"], key="iv_act")
    manual = c[3].slider("your guess for σ_w", 0.01, 0.60, 0.05, 0.01, key="iv_manual",
                         help="A 'reasonable-looking' constant, chosen without the algebra.")

    act = {"ReLU": lambda z: np.maximum(z, 0),
           "tanh": np.tanh,
           "none (linear)": lambda z: z}[act_name]

    # Plot the ROOT MEAN SQUARE, not the standard deviation. The algebra preserves the
    # second moment E[a^2]; for ReLU the two differ, because the output has a positive
    # mean (std = sqrt(1/2 - 1/2pi)*s = 0.5838*s while RMS = 0.7071*s). Plotting std put
    # correctly-initialised He at 0.826 against a "healthy = 1" line, making right look
    # 17% wrong. Averaging a few seeds removes the random-walk scatter that a single
    # draw of W shows at finite width.
    SEEDS = 5

    def run(sigma):
        acc = np.zeros(depth)
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            a = rng.normal(0, 1, (1024, width))
            for k in range(depth):
                W = rng.normal(0, sigma(width), (width, width))
                a = act(a @ W)
                acc[k] += float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        return list(acc / SEEDS)

    curves = {
        f"your guess  σ={manual:.2f}": run(lambda n: manual),
        "Xavier  √(1/n)": run(lambda n: math.sqrt(1.0 / n)),
        "He  √(2/n)": run(lambda n: math.sqrt(2.0 / n)),
    }

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for (lab, ys), col in zip(curves.items(), ("#C0507A", "#2F7BEA", "#1D9E75")):
        ax.semilogy(range(1, depth + 1), np.maximum(ys, 1e-300), "o-", ms=3.5, lw=1.9,
                    color=col, label=lab)
    ax.axhline(1.0, color="#9C9B95", ls="--", lw=1, label="healthy (RMS = 1)")
    ax.set_xlabel("layer"); ax.set_ylabel("activation RMS  √E[a²]  (log scale)")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    he_last, xa_last, my_last = curves["He  √(2/n)"][-1], curves["Xavier  √(1/n)"][-1], \
        curves[f"your guess  σ={manual:.2f}"][-1]
    m = st.columns(3)
    m[0].metric(f"your guess, layer {depth}", f"{my_last:.3g}")
    m[1].metric(f"Xavier, layer {depth}", f"{xa_last:.4f}")
    m[2].metric(f"He, layer {depth}", f"{he_last:.4f}")

    if act_name == "ReLU":
        ratio = (xa_last / curves["Xavier  √(1/n)"][0]) ** (1 / max(depth - 1, 1))
        st.success(
            f"**Predicted from the algebra:** ReLU keeps exactly half the second moment, so "
            f"Xavier's √(1/n) must lose a factor of 1/√2 = **{1/math.sqrt(2):.4f}** per layer. "
            f"**Measured here: {ratio:.4f}.** He's √(2/n) puts the missing factor of 2 back and "
            f"sits on 1 — which is all that constant ever was.",
            icon=":material/check_circle:")
        st.caption("The curve is the **RMS**, √E[a²] — the quantity the algebra actually "
                   "preserves. For ReLU that is not the standard deviation: the output has a "
                   "positive mean, so std = 0.584·s while RMS = 0.707·s. Plotting std would "
                   "park a correctly-initialised He at 0.826 and make right look 17 % wrong. "
                   f"Averaged over {SEEDS} seeds — a single draw of W random-walks by roughly "
                   "±√(2/n) per layer, which at this width is visible but is noise, not decay.")
    elif act_name == "tanh":
        st.info("**tanh is ≈ linear near zero**, so Xavier's √(1/n) is the right scale here and "
                "He's √(2/n) slightly over-drives it into the saturating region. This is why the "
                "scheme follows the nonlinearity, not taste. (tanh is symmetric, so RMS and std "
                "agree here — unlike ReLU.)", icon=":material/lightbulb:")
    else:
        st.info("**With no nonlinearity, Xavier holds the RMS exactly at 1** — that is the case "
                "the algebra was solved for. Every other curve is the correction for what the "
                "activation function does to it.", icon=":material/lightbulb:")

    st.caption("Same experiment as `experiments/tier2_training/e10`, and the constants are "
               "the ones in `core/init.py`. Full derivation in the Theory tab, §10.")


with tab_play:
    mode = st.radio("playground",
                    ["Distribution explorer", "Central Limit Theorem",
                     "🧪 Bayes' rule", "📈 Maximum likelihood",
                     "🏗 Init & variance"],
                    horizontal=True, key="p_mode")
    st.divider()
    if mode == "Distribution explorer":
        _explorer()
    elif mode == "Central Limit Theorem":
        _clt()
    elif mode.startswith("🧪"):
        _bayes()
    elif mode.startswith("🏗"):
        _init_variance()
    else:
        _mle()

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)
with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mathprob")
with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Fair die: $E[X]=3.5$; $E[X^2]=\frac{91}{6}$, so $\text{Var}=E[X^2]-E[X]^2=\frac{91}{6}-12.25=\frac{35}{12}\approx2.92$.

**2.** Base rate $1/1000$ with a 99%-accurate test: $P(\text{sick}\mid+)=\dfrac{0.99\cdot0.001}{0.99\cdot0.001+0.01\cdot0.999}\approx\mathbf{9\%}$ — still low, because rare true positives are swamped by false positives.

**3.** For Gaussian residuals, $\log\mathcal N(y\mid\hat y,\sigma^2)=-\frac{(y-\hat y)^2}{2\sigma^2}+\text{const}$; maximizing the sum $\Leftrightarrow$ minimizing $\sum(y-\hat y)^2$ — least squares **is** Gaussian MLE.

**4.** Two fair flips: each of HH, HT, TH, TT has probability $1/4=\tfrac12\cdot\tfrac12=P(A)P(B)$ → **independent**.""",
        label="Pencil & paper 1–4",
    )
    lessons.solution(
        r"""**5–7.** A histogram of `np.random.normal` overlays its PDF; averaging $k=1,2,30$ uniforms shows the **CLT** (the average → Gaussian, spread → 0); Bayes' rule reproduces the ~9% answer.

**8.** MSE ↔ **Gaussian MLE** (M1) and cross-entropy ↔ **Bernoulli/categorical MLE** (M2): the losses *are* negative log-likelihoods.""",
        label="Code & bridge 5–8",
    )
with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
