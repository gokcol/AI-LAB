"""Guided foundations shared by lesson pages and the track-aware Study Coach."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from gui import guided_content as content, practice, progress
from gui import guided_simulations as sim


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
    if topic in ("math_probability", "math_optimization", "ml_model_selection"):
        return render_next_explorer(topic)
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


def render_next_explorer(topic):
    key = "guided_explore_" + topic
    if topic == "math_probability":
        prior = st.slider("Spam base rate (%)", 0, 100, 10, key=key+"_prior")/100
        catch = st.slider("Spam caught (%)", 0, 100, 80, key=key+"_catch")/100
        false = st.slider("Legitimate messages falsely flagged (%)", 0, 100, 10, key=key+"_false")/100
        settings = {"prior": prior, "catch_rate": catch, "false_alarm_rate": false, "population": 1000}
        prompt = "Out of 1,000 messages, predict true flags, false flags and the fraction of flags that are spam. Explain the denominator."
    elif topic == "math_optimization":
        curvature = st.slider("Curvature scale c", 1, 10, 2, key=key+"_curvature")
        rate = st.slider("Learning rate η", .01, 1.1, .1, .01, key=key+"_rate")
        start = st.slider("Starting w", -5., 5., 4., .5, key=key+"_start")
        steps = st.slider("Number of updates", 2, 30, 10, key=key+"_steps")
        settings = {"curvature": curvature, "rate": round(rate, 2), "start": start, "steps": steps}
        st.write("Compare your learning rate with η = 0.25/c on L(w) = cw². Both runs start at the same w.")
        prompt = "Predict whether your run approaches zero, alternates sides, or diverges. What changes when you increase curvature?"
    else:
        mode = st.radio("Evaluation experiment", ["Accuracy and the baseline", "Validation before test"], key=key+"_mode", horizontal=True)
        if mode == "Accuracy and the baseline":
            positives = st.slider("Spam messages among 100", 1, 99, 10, key=key+"_positives")
            caught = st.slider("Spam caught (%)", 0, 100, 80, key=key+"_caught")/100
            false = st.slider("Legitimate messages falsely flagged (%)", 0, 100, 10, key=key+"_false")/100
            settings = {"mode": mode, "positives": positives, "caught": caught, "false": false}
            prompt = "Predict whether the filter beats an always-negative baseline on accuracy and recall. Which measure fits the goal of finding spam?"
        else:
            metric = st.selectbox("Choose a validation criterion", ["Accuracy", "Recall", "F1"], key=key+"_metric")
            candidate = st.selectbox("Candidate to evaluate", list(sim.VALIDATION), key=key+"_candidate")
            st.write("Validation counts (TP, FP, FN, TN): " + "; ".join(f"{name}: {counts}" for name, counts in sim.VALIDATION.items()))
            st.caption("Same 100 validation rows for every candidate. Choose the criterion first. No test counts are used to rank candidates.")
            settings = {"mode": mode, "metric": metric, "candidate": candidate}
            prompt = "Predict which candidate wins on the chosen validation criterion. Explain your choice before viewing its reserved test result."
    record = practice.prediction_gate(key, title=content.LESSONS[topic]["title"], lesson=topic,
        seed="fixed teaching example", settings=settings, prompt=prompt)
    if record is None:
        return
    if topic == "math_probability":
        result = sim.flag_counts(**settings)
        st.dataframe(pd.DataFrame({"Flagged": [result["TP"], result["FP"]],
            "Not flagged": [result["FN"], result["TN"]]}, index=["Spam", "Legitimate"]))
        st.bar_chart(pd.DataFrame({"Expected flags": [result["TP"], result["FP"]]}, index=["Spam", "Legitimate"]))
        posterior = result["posterior"]
        st.metric("Spam among flagged messages", f"{posterior:.1%}" if posterior is not None else "Undefined: no flags")
        st.write(f"True flags = {result['TP']:g}; false flags = {result['FP']:g}. Divide true flags by their sum, not by the whole inbox.")
        st.caption("Expected counts under the chosen rates; decimals are intentional. Change the base rate while keeping both conditional rates fixed.")
    elif topic == "math_optimization":
        rows, truncated = sim.descent(**settings)
        reference, _ = sim.descent(curvature, .25/curvature, start, steps)
        result = {"final": rows[-1], "stopped_at_limit": truncated, "reference_final": reference[-1]}
        frame = pd.DataFrame(rows).set_index("Step")
        chart = frame[["Loss"]].rename(columns={"Loss": "Your rate"}).reindex(range(steps+1))
        chart["Reference: 0.25/c"] = [r["Loss"] for r in reference]
        st.line_chart(chart)
        st.dataframe(frame, width="stretch")
        factor = 1-2*settings["rate"]*curvature
        st.write(f"Signed distance multiplier = **{factor:.3g}**; stable interval for a non-optimal start: **0 < η < {1/curvature:.3g}**.")
        if truncated:
            st.warning("Run stopped before loss exceeded 10¹². The missing tail is not convergence; reduce the learning rate.")
        elif start == 0:
            st.info("Already at the optimum: the gradient is zero, so no update moves w.")
        elif abs(factor) < 1:
            st.success("Distance shrinks on this quadratic. A negative multiplier means it alternates sides while converging.")
        elif abs(abs(factor)-1) < 1e-10:
            st.warning("This is the stability boundary: the distance does not shrink.")
        else:
            st.warning("Distance grows: the step is too large for this curvature.")
        st.caption("The reference rate is chosen for this toy quadratic, not a universal training recommendation. Inspect the table if a large loss hides the small curve.")
    elif settings["mode"] == "Accuracy and the baseline":
        counts = sim.flag_counts(positives/100, caught, false, 100)
        scores = sim.metrics(*(counts[k] for k in ("TP", "FP", "FN", "TN")))
        baseline = sim.metrics(0, 0, positives, 100-positives)
        result = {"expected_counts": counts, "filter": scores, "baseline": baseline}
        frame = pd.DataFrame([scores, baseline], index=["Filter", "Always negative"])
        st.bar_chart(frame[["Accuracy", "Recall", "F1"]])
        st.dataframe(frame)
        st.write("Expected confusion counts: " + ", ".join(f"{k}={counts[k]:g}" for k in ("TP", "FP", "FN", "TN")))
        st.caption("Scores are fractions. Empty precision means undefined (no positive predictions). Counts may be fractional because this is an expectation, not a random sample. Accuracy can reward missing every rare positive.")
    else:
        validation = {name: sim.metrics(*counts) for name, counts in sim.VALIDATION.items()}
        best = max(values[metric] for values in validation.values())
        winners = [name for name, values in validation.items() if values[metric] == best]
        st.dataframe(pd.DataFrame(validation).T)
        st.write(f"Best validation {metric}: **{' / '.join(winners)}**. Your candidate: **{candidate}**.")
        result = {"criterion": metric, "candidate": candidate, "validation": validation[candidate], "validation_winners": winners}
        st.info("A final test evaluates a choice already made. Repeatedly changing the candidate after seeing test results makes this set part of selection.")
        if st.button("Evaluate selected candidate on reserved test", key=key+"_test_"+record["id"]):
            st.session_state[key+"_tested_"+record["id"]] = True
        if st.session_state.get(key+"_tested_"+record["id"]):
            result["test"] = sim.metrics(*sim.TEST[candidate])
            st.dataframe(pd.DataFrame([result["validation"], result["test"]], index=["Validation", "Reserved test"]))
            st.caption("Synthetic teaching data. Once inspected, these test results are known to you; changing choices afterwards is practice, not an independent final evaluation.")
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
