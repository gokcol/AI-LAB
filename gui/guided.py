"""Guided foundations shared by lesson pages and the track-aware Study Coach."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from gui import guided_content as content, practice, progress


def render_exercises(topic, *, namespace="lesson"):
    prefix = f"guided_{namespace}_{topic}"
    summary = content.practice_summary(topic, progress.state().get("practice", {}))
    st.caption(f"{summary['solved']}/8 exercise types solved · {summary['independent']}/8 solved without hints or revealed answers. Self-assessed checkpoints stay separate.")
    st.caption("Saved in this browser with your learning file. Choose a fresh number set to practise again. Some conceptual questions intentionally repeat; this is practice, not a supervised assessment.")
    variant = st.selectbox("Practice set", range(5), format_func=lambda v: f"Set {v+1}", key=prefix+"_variant")
    questions = content.exercises(topic, variant)
    index = st.selectbox("Practice exercise", range(8), format_func=lambda i: f"{i+1}. {questions[i].title}", key=prefix+"_exercise")
    question = questions[index]
    key = content.practice_key(topic, index, variant)
    record = progress.state().get("practice", {}).get(key, {})
    widget = f"{prefix}_{index}_{variant}"
    st.markdown(f"#### {index+1}. {question.title}")
    st.write(question.prompt)
    with st.form(widget+"_answer_form"):
        if question.options:
            answer = st.radio("Your answer", question.options, index=None, key=widget+"_answer")
        else:
            answer = st.text_input("Your numeric answer", key=widget+"_answer", max_chars=40,
                                   help="Use a decimal point. Answers within 0.005 are accepted; do not enter a percentage unless requested.")
        if st.form_submit_button("Check answer"):
            if answer is None or isinstance(answer, str) and not answer.strip():
                st.warning("Enter an answer first.")
            else:
                correct = question.check(answer)
                progress.replace(progress.record_practice(progress.state(), key, "check", correct=correct))
                st.session_state[widget+"_feedback"] = (correct, "Correct. " + question.solution if correct else question.feedback(answer))
                st.rerun()
    feedback = st.session_state.get(widget+"_feedback")
    if feedback:
        (st.success if feedback[0] else st.warning)(feedback[1])
    if record.get("independent"):
        st.caption("Recorded: solved without hints or a revealed answer.")
    elif record.get("solved"):
        st.caption("Recorded: solved with help. Try another set without hints when you are ready.")
    hints = record.get("hints", 0)
    if hints < len(question.hints) and st.button("Show next hint", key=widget+"_hint"):
        progress.replace(progress.record_practice(progress.state(), key, "hint"))
        st.rerun()
    for n, hint in enumerate(question.hints[:hints], 1):
        st.info(f"Hint {n}: {hint}")
    if not record.get("revealed") and st.button("Reveal worked answer", key=widget+"_reveal"):
        progress.replace(progress.record_practice(progress.state(), key, "reveal"))
        st.rerun()
    if record.get("revealed"):
        st.write("**Worked answer:** " + question.solution)
        st.caption("Revealing an answer does not mark it solved. Later checks on this set are recorded as assisted.")


def render_explorer(topic):
    key = "guided_explore_" + topic
    if topic == "math_vectors":
        a, b = st.columns(2)
        x1 = a.slider("First feature x₁", -4, 6, 3, key=key+"_x1")
        x2 = b.slider("Second feature x₂", -4, 6, 4, key=key+"_x2")
        settings = {"x1": x1, "x2": x2, "w1": 2, "w2": -1, "bias": 1}
        prompt = "With weights (2, −1) and bias 1, predict the score. Which feature contributes more?"
    elif topic == "math_calculus":
        w = st.slider("Starting w", -3., 5., 3., .5, key=key+"_w")
        rate = st.select_slider("Learning rate η", [.1, .25, .5, 1., 1.5], value=.25, key=key+"_rate")
        settings = {"w": w, "rate": rate, "target": 1}
        prompt = "Predict the new w and whether L = (w − 1)² will decrease after one update. Explain your prediction."
    elif topic == "ml_regression":
        w = st.slider("Slope w", -1., 4., 1., .5, key=key+"_w")
        b = st.slider("Intercept b", -2., 4., 0., .5, key=key+"_b")
        st.write("Known rows: x = [0, 1, 2, 3], y = [1, 3, 5, 7].")
        settings = {"w": w, "b": b}
        prompt = "Predict the four outputs and whether this line beats the constant prediction 4 on these rows."
    else:
        threshold = st.slider("Decision threshold τ", 0., 1., .5, .1, key=key+"_threshold")
        st.write("Fixed scores: [0.9, 0.6, 0.4, 0.2]. True labels: [1, 0, 1, 0]. Positive means score ≥ threshold.")
        settings = {"threshold": round(threshold, 2)}
        prompt = "Predict the positive labels, false positives and missed positives at this threshold."
    record = practice.prediction_gate(key, title=content.LESSONS[topic]["title"], lesson=topic,
        seed="fixed teaching example", settings=settings, prompt=prompt)
    if record is None:
        return
    if topic == "math_vectors":
        contributions = [2*x1, -x2]
        result = {"contributions": contributions, "dot_product": sum(contributions), "score_with_bias": sum(contributions)+1}
        st.bar_chart(pd.DataFrame({"Contribution": contributions}, index=["2 × x₁", "−1 × x₂"]))
        st.metric("Score including bias", result["score_with_bias"])
        st.write(f"{2*x1} + ({-x2}) + 1 = {result['score_with_bias']}. Negative bars subtract from the score.")
    elif topic == "math_calculus":
        gradient = 2*(w-1)
        new = w-rate*gradient
        result = {"gradient": gradient, "old_w": w, "new_w": new, "old_loss": (w-1)**2, "new_loss": (new-1)**2}
        grid = np.linspace(min(-3, new-1), max(5, new+1), 100)
        st.line_chart(pd.DataFrame({"w": grid, "Loss": (grid-1)**2}), x="w", y="Loss")
        st.dataframe(pd.DataFrame({"Step": ["Before", "After"], "w": [w, new], "Loss": [(w-1)**2, (new-1)**2]}), hide_index=True)
        st.write(f"Gradient = {gradient:g}; new w = {w:g} − {rate:g} × ({gradient:g}) = {new:g}.")
    elif topic == "ml_regression":
        x = np.arange(4)
        y = np.array([1, 3, 5, 7])
        pred = w*x+b
        error = pred-y
        result = {"predictions": pred.tolist(), "MSE": float(np.mean(error**2)), "baseline_MSE": 5.}
        st.line_chart(pd.DataFrame({"x": x, "Observed": y, "Predicted": pred, "Constant baseline": np.full(4, 4)}), x="x")
        st.dataframe(pd.DataFrame({"x": x, "Observed": y, "Prediction": pred, "Error": error, "Squared error": error**2}), hide_index=True)
        st.write(f"Model MSE = **{result['MSE']:g}**; constant-4 baseline MSE = **5**. Both are measured on these same four teaching rows.")
    else:
        scores = np.array([.9, .6, .4, .2])
        actual = np.array([1, 0, 1, 0])
        predicted = (scores >= settings["threshold"]).astype(int)
        tp = int(np.sum((predicted == 1) & (actual == 1)))
        fp = int(np.sum((predicted == 1) & (actual == 0)))
        fn = 2-tp
        tn = 2-fp
        precision = tp/(tp+fp) if tp+fp else None
        result = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": tp/2}
        st.dataframe(pd.DataFrame({"Score": scores, "True label": actual, "Predicted label": predicted}), hide_index=True)
        st.dataframe(pd.DataFrame({"Predicted 1": [tp, fp], "Predicted 0": [fn, tn]}, index=["Actual 1", "Actual 0"]))
        a, b = st.columns(2)
        a.metric("Precision", f"{precision:.2f}" if precision is not None else "Undefined")
        b.metric("Recall", f"{tp/2:.2f}")
        st.caption("Rows are actual labels; columns are predictions. Changing the threshold changes decisions, not the fixed scores.")
    practice.save_result(record, json.dumps(result))


def render(topic):
    lesson = content.LESSONS[topic]
    st.subheader(lesson["title"])
    st.caption("A guided foundation · 15–25 minutes · no code setup needed")
    step = st.segmented_control("Guided step", ["Understand", "Worked example", "Predict & explore", "Practice", "Connect"],
                                default="Understand", key="guided_step_"+topic) or "Understand"
    if step == "Understand":
        st.write(lesson["intro"])
        st.latex(lesson["formula"])
        st.write(lesson["symbols"])
        st.info(lesson["scope"])
        st.write("Next: open **Worked example**, then make your own prediction in **Predict & explore**.")
    elif step == "Worked example":
        st.markdown(lesson["worked"])
        st.write("Next: change a setting in **Predict & explore**. Record your prediction before seeing the result.")
    elif step == "Predict & explore":
        render_explorer(topic)
    elif step == "Practice":
        render_exercises(topic)
    else:
        st.write(lesson["bridge"])
        st.markdown(lesson["derive"])
        for route, label in lesson["links"]:
            st.page_link(f"views/{route}.py", label=label)
        st.page_link("views/study_coach.py", label="Ask the Study Coach what to practise next")
        st.caption("Explain the connection in your checkpoint below. Reading a solution does not automatically complete a practice exercise.")
