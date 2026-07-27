import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lessons
import ml_lessons
import training_ui
from core import training as training_core

LESSON = ml_lessons.LINEAR_REGRESSION


def _gen(n, tw, tb, noise, seed):
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-3.0, 3.0, int(n))
    y = tw * x + tb + rng.normal(0.0, noise, int(n))
    return x, y


def _ols(x, y):
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), float(beta[0])  # (w_hat, b_hat)


def _fit_callback():
    s = st.session_state
    x, y = _gen(s.lr_n, s.lr_tw, s.lr_tb, s.lr_noise, s.lr_seed)
    w_hat, b_hat = _ols(x, y)
    s.lr_w = float(np.clip(round(w_hat, 1), -5.0, 5.0))
    s.lr_b = float(np.clip(round(b_hat, 1), -5.0, 5.0))


st.title("Linear regression — playground & lesson")
st.caption("M1 · ŷ = w·x + b.  Fit a line to data by hand, then let least squares do it perfectly.")

lessons.predict(
    "You'll fit a line by hand, then press **Fit**. What does 'least squares' actually minimize — and is there more than one best line?",
    "It minimizes the **sum of squared residuals** (vertical distances). For a straight line that surface is a bowl with a *single* minimum, so the best line is **unique** — and there's a closed-form (OLS) solution, no search needed. Geometrically ŷ is the orthogonal projection of y onto the inputs' span.",
)

tab_play, tab_train, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🎛 Playground", "🏋 Train step by step", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_play:
    # Model sliders are also driven by the Fit button via session_state, so seed
    # defaults here and omit the value= arg (avoids a Streamlit double-set warning).
    st.session_state.setdefault("lr_w", 1.0)
    st.session_state.setdefault("lr_b", 0.0)

    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("**Data**")
        n = st.slider("points", 10, 200, 40, key="lr_n")
        tw = st.slider("true slope", -3.0, 3.0, 1.5, 0.1, key="lr_tw")
        tb = st.slider("true intercept", -3.0, 3.0, 0.5, 0.1, key="lr_tb")
        noise = st.slider("noise σ", 0.0, 3.0, 1.0, 0.1, key="lr_noise")
        seed = st.number_input("seed", 0, 9999, 0, key="lr_seed")
        st.markdown("**Your model**")
        w = st.slider("w (slope)", -5.0, 5.0, step=0.1, key="lr_w")
        b = st.slider("b (intercept)", -5.0, 5.0, step=0.1, key="lr_b")
        show_resid = st.checkbox("show residuals", value=True)
        st.button("Fit (least squares)", icon=":material/auto_fix_high:",
                  type="primary", on_click=_fit_callback)

    x, y = _gen(n, tw, tb, noise, seed)
    w_hat, b_hat = _ols(x, y)

    yhat_user = w * x + b
    mse_user = float(np.mean((y - yhat_user) ** 2))
    mse_ols = float(np.mean((y - (w_hat * x + b_hat)) ** 2))
    ss_res = float(np.sum((y - yhat_user) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_user = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    with right:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        xs = np.array([x.min(), x.max()])
        if show_resid:
            ax.vlines(x, np.minimum(y, yhat_user), np.maximum(y, yhat_user),
                      color="#C0507A", alpha=0.25, lw=1, zorder=1)
        ax.scatter(x, y, s=28, color="#5B8FC2", edgecolors="white", linewidths=0.5,
                   zorder=2, label="data")
        ax.plot(xs, w_hat * xs + b_hat, color="#1D9E75", lw=2, ls="--",
                zorder=3, label="least-squares fit")
        ax.plot(xs, w * xs + b, color="#C0507A", lw=2.5, zorder=4, label="your model")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("fit a line to the data")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Your MSE", f"{mse_user:.3f}")
    c2.metric("Best MSE (OLS)", f"{mse_ols:.3f}")
    c3.metric("Your R²", f"{r2_user:.3f}")
    st.latex(rf"\hat y = {w:.1f}\,x + ({b:.1f}) \qquad \text{{(least squares: }} "
             rf"\hat y = {w_hat:.2f}\,x + ({b_hat:.2f}))")
    st.info("Minimize **Your MSE** by hand, then press **Fit** to jump to the optimum. "
            "Open **Theory** for why least squares is that optimum.", icon=":material/lightbulb:")

with tab_train:
    st.markdown("### The story")
    st.info("You run a small delivery desk. A customer asks, “If the address is this far away, "
            "when should I expect the parcel?” You have past trips—distance and actual minutes—"
            "but no trustworthy rule yet. The model starts with a bad guess, then learns a fair "
            "minutes-per-kilometre estimate from those trips.")
    st.markdown("Train a delivery-time predictor from simulated data. The data are fake, but the "
                "calculation is the same one used for a real one-feature regression model.")
    controls = st.columns(4)
    n_train = controls[0].select_slider("deliveries", [16, 24, 32, 48], value=32, key="tr_reg_n")
    noise_train = controls[1].select_slider("measurement noise", [0.0, 1.0, 2.0, 3.0], value=2.0,
                                             key="tr_reg_noise")
    lr_train = controls[2].select_slider("learning rate η", [0.001, 0.002, 0.003, 0.004, 0.006], value=.003,
                                          key="tr_reg_lr")
    seed_train = controls[3].number_input("data seed", 0, 9999, 4, key="tr_reg_seed")
    data = training_core.delivery_data(int(n_train), float(noise_train), int(seed_train))
    with st.expander("Peek at the dataset (first 16 rows)"):
        preview_data = pd.concat([
            pd.DataFrame({"split": "train", "distance (km)": data.raw_train[:, 0],
                          "actual minutes": data.y_train}),
            pd.DataFrame({"split": "held-out", "distance (km)": data.raw_test[:, 0],
                          "actual minutes": data.y_test}),
        ], ignore_index=True)
        st.dataframe(preview_data.head(16).round(3), hide_index=True, width="stretch")
        st.caption("Only the training rows contribute to gradients. Held-out rows are used after "
                   "training to check whether the line generalizes.")
    focus_index = st.selectbox(
        "Focus on one training delivery", range(len(data.X_train)), key="tr_reg_focus",
        format_func=lambda i: f"delivery {i + 1}: {data.X_train[i, 0]:.1f} km → {data.y_train[i]:.1f} min",
    )
    with st.expander("Reveal how this dataset was generated"):
        st.latex(r"\text{minutes}=8+2.2\times\text{distance}+\text{random noise}")
        st.caption("The learner sees this only after experimenting; OLS and gradient descent should "
                   "both approach the hidden relationship when enough data are available.")
    config = (int(n_train), float(noise_train), float(lr_train), int(seed_train))
    state_key = "training_regression"
    state = st.session_state.get(state_key)
    if state is None or state["config"] != config:
        state = {"config": config, "parameters": np.zeros(2), "history": [], "epoch": 0, "diverged": False}
        st.session_state[state_key] = state

    action = training_ui.action_bar("tr_reg")
    if action == "reset":
        state = {"config": config, "parameters": np.zeros(2), "history": [], "epoch": 0, "diverged": False}
    elif action:
        count = {"one": 1, "epoch": 1, "ten": 10, "all": 1500}[action]
        for _ in range(count):
            step = training_core.regression_step(data.X_train, data.y_train, state["parameters"],
                                                 lr=float(lr_train), epoch=state["epoch"])
            state["history"].append(step)
            state["parameters"] = step.parameters_after
            state["epoch"] += 1
            if not np.isfinite(step.loss) or step.loss > 1e12:
                state["diverged"] = True
                break
    st.session_state[state_key] = state
    if not state["history"]:
        preview = training_core.regression_step(data.X_train, data.y_train, state["parameters"],
                                                lr=float(lr_train))
        st.info("Start from random-free weights `w = 0, b = 0`, then take one step to make the "
                "first update part of the replay history.")
        selected = preview
    else:
        selected = training_ui.checkpoint_picker("tr_reg", state["history"])

    phase = training_ui.phase_picker("tr_reg")
    x = data.X_train[:, 0]
    residual = selected.predictions - data.y_train
    if phase == "1 · Forward":
        st.latex(r"\hat y_i = w x_i + b")
        st.dataframe(pd.DataFrame({"focus": ["← chosen" if i == focus_index else "" for i in range(len(x))],
                                   "distance (km)": x, "actual minutes": data.y_train,
                                   "prediction": selected.predictions}).round(4),
                     hide_index=True, width="stretch")
    elif phase == "2 · Loss":
        st.latex(r"L = \frac{1}{n}\sum_i(\hat y_i-y_i)^2")
        st.dataframe(pd.DataFrame({"focus": ["← chosen" if i == focus_index else "" for i in range(len(x))],
                                   "residual": residual, "squared residual": selected.per_example_loss}).round(5),
                     hide_index=True, width="stretch")
        st.metric("mean squared error", f"{selected.loss:.6f}")
    elif phase == "3 · Backward":
        st.latex(r"\frac{\partial L}{\partial w}=\frac{2}{n}\sum_i x_i(\hat y_i-y_i),\quad "
                 r"\frac{\partial L}{\partial b}=\frac{2}{n}\sum_i(\hat y_i-y_i)")
        st.dataframe(pd.DataFrame({"x × residual": x * residual,
                                   "residual": residual}).round(5), hide_index=True, width="stretch")
        st.write(f"`dw = {selected.gradients[0]:.6f}`, `db = {selected.gradients[1]:.6f}`")
    elif phase == "4 · Update":
        training_ui.parameter_ledger(("slope w", "intercept b"), selected)
        direction = "increased" if selected.delta[0] > 0 else "decreased"
        st.info(f"**What changed?** The slope {direction} because the batch's weighted residual "
                f"was {selected.gradients[0]:.3f}. The chosen delivery contributes "
                f"{x[focus_index] * residual[focus_index]:.3f} to that slope gradient.")
        training_ui.check_number("tr_reg_w", "Your new slope w", selected.parameters_after[0])
    else:
        current = selected.parameters_after
        _, _, train_mse, _ = training_core.regression_values(data.X_train, data.y_train, current)
        _, _, test_mse, _ = training_core.regression_values(data.X_test, data.y_test, current)
        ols_w, ols_b = _ols(data.X_train[:, 0], data.y_train)
        baseline = float(np.mean(data.y_train))
        baseline_test = float(np.mean((data.y_test - baseline) ** 2))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("training MSE", f"{train_mse:.3f}")
        c2.metric("held-out MSE", f"{test_mse:.3f}")
        c3.metric("distance to OLS", f"{np.hypot(current[0]-ols_w, current[1]-ols_b):.3f}")
        c4.metric("do-nothing baseline MSE", f"{baseline_test:.3f}",
                  help="Always predict the average training delivery time.")
        hardest = int(np.argmax(np.abs(data.y_test - (current[0] * data.X_test[:, 0] + current[1]))))
        st.info(f"**A surprising held-out delivery:** {data.X_test[hardest, 0]:.1f} km actually took "
                f"{data.y_test[hardest]:.1f} min. The model predicts "
                f"{current[0] * data.X_test[hardest, 0] + current[1]:.1f} min—its largest test residual.")
        if state.get("diverged"):
            st.warning("This learning rate overshot the MSE bowl and diverged. Reset, then choose "
                       "a smaller η to trade speed for stability.")
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        ax.scatter(data.X_train[:, 0], data.y_train, color="#185FA5", label="train")
        ax.scatter(data.X_test[:, 0], data.y_test, color="#A32D2D", marker="^", label="held-out")
        grid = np.linspace(0, 21, 80)
        ax.plot(grid, current[0] * grid + current[1], color="#1D9E75", label="current model")
        ax.plot(grid, ols_w * grid + ols_b, "--", color="#9C6B2F", label="OLS benchmark")
        ax.set(xlabel="distance (km)", ylabel="minutes"); ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")

    with st.expander("Compare two learning rates"):
        compare_lr = st.select_slider("comparison η", [0.001, 0.002, 0.003, 0.004, 0.006], value=.006,
                                      key="tr_reg_compare_lr")
        def _losses(rate):
            p, out = np.zeros(2), []
            for epoch in range(80):
                s = training_core.regression_step(data.X_train, data.y_train, p, lr=rate, epoch=epoch)
                out.append(min(s.loss, 1e12)); p = s.parameters_after
                if not np.isfinite(s.loss) or s.loss > 1e12:
                    break
            return out
        st.line_chart(pd.DataFrame({f"current η={lr_train}": pd.Series(_losses(float(lr_train))),
                                    f"comparison η={compare_lr}": pd.Series(_losses(float(compare_lr)))}),
                      height=220)
        st.caption("Compare progress, not just the final point: a large step can learn quickly, "
                   "oscillate, or overshoot the bowl entirely.")

    numeric = training_core.finite_difference(
        lambda p: training_core.regression_values(data.X_train, data.y_train, p)[2],
        selected.parameters_before,
    )
    training_ui.verification_panel(selected.loss, selected.gradients, numeric,
                                   ("Held-out delivery rows are never used in the update.",
                                    "The dashed line is an independent OLS benchmark."))

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="linreg")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Your eyeball fit lands close but rarely beats **Fit** — least squares is the exact minimum of the MSE bowl.

**2.** At σ = 0 the points lie exactly on a line, so $R^2 = 1$ (all variance explained). As noise rises, more variance is unexplained and $R^2$ falls.

**3.** Few points + high noise + different seeds → the fitted line **swings** a lot: small samples give high-variance estimates.""",
        label="Warm-up 1–3",
    )
    lessons.solution(
        r"""**4.** MSE $=\frac1n\lVert y - X\boldsymbol\beta\rVert^2$. Set $\partial/\partial\boldsymbol\beta = -\frac2n X^\top(y - X\boldsymbol\beta) = 0 \Rightarrow X^\top X\boldsymbol\beta = X^\top y$ — the **normal equations**.

**5.** One feature: minimizing $\sum (y_i-(wx_i+b))^2$ gives $b=\bar y - w\bar x$ and $w=\operatorname{cov}(x,y)/\operatorname{var}(x)$.

**6.** $\hat y = X\boldsymbol\beta$ ranges over $\text{span}(X)$; least squares picks the point in that span closest to $y$, so the residual $y-\hat y$ is **perpendicular** to $\text{span}(X)$ — $\hat y$ is the orthogonal projection of $y$.""",
        label="Pencil & paper 4–6",
    )
    lessons.solution(
        r"""**7–8.** `np.linalg.lstsq` and $(X^\top X)^{-1}X^\top y$ agree; gradient descent on the MSE converges to that same OLS solution.

**9–10.** Features $[x,x^2,x^3]$ fit curves with the *same linear* solver (it's linear in the weights); a ridge penalty $\lambda$ shrinks every weight toward 0 as $\lambda$ grows.

**11.** A **linear neuron + MSE** (ANN Tier 1) trained on the same data converges to the OLS line — where the two modules meet.""",
        label="Code & bridge 7–11",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
