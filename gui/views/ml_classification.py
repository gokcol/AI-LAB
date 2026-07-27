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

LESSON = ml_lessons.CLASSIFICATION


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logreg(X, y, lr=0.3, steps=800):
    """Logistic regression via gradient descent; returns a proba(X) function."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = _sigmoid(Xs @ w + b)
        w -= lr * (Xs.T @ (p - y)) / len(y)
        b -= lr * (p - y).mean()
    return lambda Xn: _sigmoid(((np.asarray(Xn) - mu) / sd) @ w + b)


st.title("M2 · Classification — playground & lesson")
st.caption("Logistic regression = a sigmoid neuron. Move the threshold and watch the "
           "confusion matrix and precision/recall trade off.")

lessons.predict(
    "Slide the decision **threshold** up. Which rises and which falls — precision or recall — and why can't you max both?",
    "Raising the threshold labels fewer points positive, so **recall cannot increase**. Precision often rises because false alarms are removed, but on a finite dataset it can wobble either way when individual points drop out. The model is fixed — the threshold slides you along its precision–recall curve.",
)

tab_train, tab_play, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🏋 Train step by step", "🎛 Evaluate thresholds", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_train:
    st.markdown("### The story")
    st.info("You are reviewing a factory shift. For each machine, you see its temperature and "
            "vibration, then decide whether it needs an inspection. Past inspections are imperfect: "
            "some hot, shaky machines survive the day and some quiet machines still fail. The model "
            "learns a probability, not a certainty—and you choose the alert threshold afterward.")
    st.markdown("Train a logistic classifier on simulated machine sensors. Temperature and vibration "
                "matter, but labels include noise—so a useful model is not a perfect one.")
    controls = st.columns(3)
    n_train = controls[0].select_slider("machines", [48, 72, 96, 120], value=96, key="tr_clf_n")
    lr_train = controls[1].select_slider("learning rate η", [.05, .10, .15, .25], value=.15,
                                          key="tr_clf_lr")
    seed_train = controls[2].number_input("data seed", 0, 9999, 4, key="tr_clf_seed")
    data = training_core.machine_data(int(n_train), int(seed_train))
    with st.expander("Peek at the dataset (first 16 rows)"):
        preview_data = pd.concat([
            pd.DataFrame({"split": "train", "temperature (°C)": data.raw_train[:, 0],
                          "vibration": data.raw_train[:, 1], "failure label": data.y_train}),
            pd.DataFrame({"split": "held-out", "temperature (°C)": data.raw_test[:, 0],
                          "vibration": data.raw_test[:, 1], "failure label": data.y_test}),
        ], ignore_index=True)
        st.dataframe(preview_data.head(16).round(3), hide_index=True, width="stretch")
        st.caption("The model sees standardized sensor values during training; raw values stay here "
                   "so the scenario remains interpretable. Held-out machines are evaluation only.")
    focus_index = st.selectbox(
        "Focus on one training machine", range(len(data.X_train)), key="tr_clf_focus",
        format_func=lambda i: f"machine {i + 1}: {data.raw_train[i, 0]:.1f}°C, vibration {data.raw_train[i, 1]:.2f}",
    )
    with st.expander("Reveal how failure labels were generated"):
        st.latex(r"P(\mathrm{failure})=\sigma(-0.9+1.0\,T_z+1.35\,V_z)")
        st.caption("A random draw turns this probability into each observed label. That is why "
                   "some apparently calm machines fail and some risky ones do not.")
    config = (int(n_train), float(lr_train), int(seed_train))
    state_key = "training_classifier"
    state = st.session_state.get(state_key)
    if state is None or state["config"] != config:
        state = {"config": config, "parameters": np.zeros(3), "history": [], "epoch": 0}
        st.session_state[state_key] = state
    action = training_ui.action_bar("tr_clf")
    if action == "reset":
        state = {"config": config, "parameters": np.zeros(3), "history": [], "epoch": 0}
    elif action:
        count = {"one": 1, "epoch": 1, "ten": 10, "all": 180}[action]
        for _ in range(count):
            step = training_core.logistic_step(data.X_train, data.y_train, state["parameters"],
                                               lr=float(lr_train), epoch=state["epoch"])
            state["history"].append(step)
            state["parameters"] = step.parameters_after
            state["epoch"] += 1
    st.session_state[state_key] = state
    if not state["history"]:
        selected = training_core.logistic_step(data.X_train, data.y_train, state["parameters"],
                                               lr=float(lr_train))
        st.info("The model begins with all weights at zero: every machine initially receives the "
                "same 50% failure probability.")
    else:
        selected = training_ui.checkpoint_picker("tr_clf", state["history"])
    phase = training_ui.phase_picker("tr_clf")
    logits, probability, terms, _, _ = training_core.logistic_values(
        data.X_train, data.y_train, selected.parameters_before
    )
    if phase == "1 · Forward":
        st.latex(r"z=w_Tx_T+w_Vx_V+b,\qquad p=\sigma(z)")
        st.dataframe(pd.DataFrame({"focus": ["← chosen" if i == focus_index else "" for i in range(len(data.X_train))],
                                   "temperature z": data.X_train[:, 0],
                                   "vibration z": data.X_train[:, 1], "logit z": logits,
                                   "P(likely failure)": probability, "target": data.y_train}).round(4),
                     hide_index=True, width="stretch")
    elif phase == "2 · Loss":
        st.latex(r"L_i=-[y_i\log p_i+(1-y_i)\log(1-p_i)]")
        st.dataframe(pd.DataFrame({"focus": ["← chosen" if i == focus_index else "" for i in range(len(data.X_train))],
                                   "target": data.y_train, "probability": probability,
                                   "BCE contribution": terms}).round(5), hide_index=True, width="stretch")
        st.metric("mean binary cross-entropy", f"{selected.loss:.6f}")
    elif phase == "3 · Backward":
        residual = probability - data.y_train
        st.latex(r"\frac{\partial L_i}{\partial z_i}=p_i-y_i")
        st.dataframe(pd.DataFrame({"p − y": residual,
                                   "temperature contribution": data.X_train[:, 0] * residual,
                                   "vibration contribution": data.X_train[:, 1] * residual}).round(5),
                     hide_index=True, width="stretch")
        st.write("`dw_temperature = %.6f`, `dw_vibration = %.6f`, `db = %.6f`" % tuple(selected.gradients))
    elif phase == "4 · Update":
        training_ui.parameter_ledger(("temperature weight", "vibration weight", "bias"), selected)
        sensor = "temperature" if abs(selected.gradients[0]) >= abs(selected.gradients[1]) else "vibration"
        direction = "up" if selected.delta[0 if sensor == "temperature" else 1] > 0 else "down"
        st.info(f"**What changed?** This batch pushed the {sensor} weight {direction}. The focused "
                f"machine has `p − y = {probability[focus_index] - data.y_train[focus_index]:.3f}`, "
                "which is its direct error signal before feature weighting.")
        training_ui.check_number("tr_clf_b", "Your new bias", selected.parameters_after[-1])
    else:
        current = selected.parameters_after
        _, p_train, _, train_bce, _ = training_core.logistic_values(data.X_train, data.y_train, current)
        _, p_test, _, test_bce, _ = training_core.logistic_values(data.X_test, data.y_test, current)
        yhat_train, yhat_test = (p_train >= .5), (p_test >= .5)
        majority = int(data.y_train.mean() >= .5)
        baseline_accuracy = float((data.y_test == majority).mean())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("training BCE", f"{train_bce:.3f}")
        c2.metric("held-out BCE", f"{test_bce:.3f}")
        c3.metric("training accuracy", f"{(yhat_train == data.y_train).mean():.2f}")
        c4.metric("held-out accuracy", f"{(yhat_test == data.y_test).mean():.2f}")
        c5.metric("majority baseline", f"{baseline_accuracy:.2f}",
                  help="Always predict the most common training label.")
        mistakes = np.flatnonzero(yhat_test != data.y_test)
        if len(mistakes):
            hard = int(mistakes[np.argmin(np.abs(p_test[mistakes] - .5))])
            st.info(f"**A surprising held-out machine:** {data.raw_test[hard, 0]:.1f}°C and "
                    f"vibration {data.raw_test[hard, 1]:.2f} looked {p_test[hard]:.0%} likely to fail, "
                    f"but its observed label was {int(data.y_test[hard])}. This is label noise or "
                    "a missing factor—not necessarily a training mistake.")
        raw = np.vstack((data.raw_train, data.raw_test))
        lim_t = np.linspace(raw[:, 0].min() - 3, raw[:, 0].max() + 3, 80)
        lim_v = np.linspace(raw[:, 1].min() - .08, raw[:, 1].max() + .08, 80)
        tt, vv = np.meshgrid(lim_t, lim_v)
        mu, sd = data.raw_train.mean(0), data.raw_train.std(0) + 1e-12
        grid = (np.c_[tt.ravel(), vv.ravel()] - mu) / sd
        zz = training_core.sigmoid(grid @ current[:-1] + current[-1]).reshape(tt.shape)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.contourf(tt, vv, zz, levels=16, cmap="RdYlBu_r", alpha=.55, vmin=0, vmax=1)
        ax.contour(tt, vv, zz, levels=[.5], colors="black", linewidths=1.4)
        for target, colour, label in [(0, "#185FA5", "normal"), (1, "#A32D2D", "likely failure")]:
            mask = data.y_train == target
            ax.scatter(data.raw_train[mask, 0], data.raw_train[mask, 1], color=colour,
                       edgecolor="white", s=28, label=f"train: {label}")
            mask = data.y_test == target
            ax.scatter(data.raw_test[mask, 0], data.raw_test[mask, 1], color=colour,
                       edgecolor="black", marker="^", s=35, label=f"test: {label}")
        ax.set(xlabel="temperature (°C)", ylabel="vibration", title="Probability of likely failure")
        ax.legend(fontsize=7, ncol=2); st.pyplot(fig, width="stretch")

    with st.expander("Compare two learning rates"):
        compare_lr = st.select_slider("comparison η", [.05, .10, .15, .25], value=.25,
                                      key="tr_clf_compare_lr")
        def _losses(rate):
            p, out = np.zeros(3), []
            for epoch in range(100):
                s = training_core.logistic_step(data.X_train, data.y_train, p, lr=float(rate), epoch=epoch)
                out.append(s.loss); p = s.parameters_after
            return out
        st.line_chart(pd.DataFrame({f"current η={lr_train}": _losses(lr_train),
                                    f"comparison η={compare_lr}": _losses(compare_lr)}), height=220)
        st.caption("On this scaled dataset both rates can learn; compare how quickly their loss "
                   "falls rather than declaring a universally best learning rate.")

    numeric = training_core.finite_difference(
        lambda p: training_core.logistic_values(data.X_train, data.y_train, p)[3],
        selected.parameters_before,
    )
    training_ui.verification_panel(selected.loss, selected.gradients, numeric,
                                   ("Standardization was fitted on training machines only.",
                                    "Held-out machines never contribute to a gradient."))

with tab_play:
    left, right = st.columns([0.42, 0.58])
    with left:
        n = st.slider("points per class", 20, 200, 60, key="c_n")
        sep = st.slider("class separation", 0.2, 3.0, 1.4, 0.1, key="c_sep")
        spread = st.slider("spread / overlap (σ)", 0.3, 2.5, 1.0, 0.1, key="c_spread")
        seed = st.number_input("seed", 0, 9999, 0, key="c_seed")
        t = st.slider("decision threshold", 0.05, 0.95, 0.50, 0.05, key="c_thr")

    rng = np.random.default_rng(int(seed))
    X0 = rng.normal([-sep, -sep], spread, (int(n), 2))
    X1 = rng.normal([sep, sep], spread, (int(n), 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(int(n)), np.ones(int(n))]

    proba = _fit_logreg(X, y)
    p = proba(X)
    pred = (p >= t).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    TN = int(((pred == 0) & (y == 0)).sum())
    acc = (TP + TN) / len(y)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    with right:
        lim = float(np.abs(X).max()) + 0.5
        gx = np.linspace(-lim, lim, 200)
        xx, yy = np.meshgrid(gx, gx)
        zz = proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(4.7, 4.3))
        cf = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[t], colors="k", linewidths=1.8)  # boundary at p=t
        ax.scatter(X0[:, 0], X0[:, 1], s=20, color="#A32D2D", edgecolors="white",
                   linewidths=0.4, label="class 0", zorder=3)
        ax.scatter(X1[:, 0], X1[:, 1], s=20, color="#185FA5", edgecolors="white",
                   linewidths=0.4, label="class 1", zorder=3)
        ax.set_title(f"P(class 1) + boundary at p={t:.2f}")
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(cf, ax=ax, shrink=0.8, label="p")
        st.pyplot(fig, width="stretch")

    m = st.columns(4)
    m[0].metric("Accuracy", f"{acc:.2f}")
    m[1].metric("Precision", f"{prec:.2f}")
    m[2].metric("Recall", f"{rec:.2f}")
    m[3].metric("F1", f"{f1:.2f}")

    cm = pd.DataFrame([[TN, FP], [FN, TP]],
                      index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"])
    st.markdown("**Confusion matrix**")
    st.dataframe(cm, width="content")
    st.info("Lowering the threshold calls more points **class 1**, so recall cannot fall. "
            "Precision often falls, but is not mathematically monotonic on finite data. "
            "A spam filter usually prioritizes precision; a cancer screen prioritizes recall.",
            icon=":material/lightbulb:")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix="mlclf")

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**1.** Lowering the threshold labels more points positive, so **recall cannot decrease**. Precision often decreases, but may wobble on a finite sample. A **spam filter** wants high precision (don't junk real mail) → usually a higher threshold; a **cancer screen** wants high recall (don't miss a case) → a lower threshold.

**2.** More overlap → accuracy drops; the optimal boundary is still a **straight line** (logistic regression is linear) — it just can't cleanly split the mixed region.

**3.** The max-F1 threshold is usually **not** 0.5; it shifts with class balance and overlap.""",
        label="Warm-up 1–3",
    )
    lessons.solution(
        r"""**4.** TP=40, FN=10, FP=5, TN=45. Accuracy $=85/100=0.85$; Precision $=40/45\approx0.889$; Recall $=40/50=0.80$; F1 $=2\cdot\frac{0.889\cdot0.80}{0.889+0.80}\approx0.842$.

**5.** Linear regression on 0/1 labels isn't bounded to $[0,1]$, and far points tilt the line, sliding the 0.5-crossing — fragile. A sigmoid squashes to a probability and is robust.

**6.** With $p=\sigma(z)$ and BCE $=-[y\log p+(1-y)\log(1-p)]$: $\frac{\partial\text{BCE}}{\partial p}=\frac{p-y}{p(1-p)}$ and $\frac{\partial p}{\partial z}=p(1-p)$, so $\frac{\partial\text{BCE}}{\partial z}=p-y$.""",
        label="Pencil & paper 4–6",
    )
    lessons.solution(
        r"""**7–9.** A NumPy logistic-regression (GD) reproduces the Playground boundary; sweeping the threshold 0→1 traces the **ROC** (and its AUC); `sklearn.linear_model.LogisticRegression` matches.

**10.** A **sigmoid neuron + BCE** *is* logistic regression — train that neuron (ANN Tier 1) on the same two-class data for the same boundary.""",
        label="Code & bridge 7–10",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
