"""Prediction-first challenges; diagnoses remain hidden until explicitly requested."""
import numpy as np
import pandas as pd
import streamlit as st
from core.challenges import learning_rate_run, leakage_run
from gui.practice import prediction_gate, save_result, hints

st.title("Debugging challenges")
st.caption("Goal: diagnose a failed experiment using evidence. Prerequisites: gradients and train/test splits. About 20 minutes. Updated 2026-09-20.")
cases = ["Loss explodes", "Perfect accuracy, wrong split", "The sigmoid barely learns", "The gradient has the wrong sign"]
case = st.selectbox("Case", range(len(cases)), format_func=lambda i: cases[i])
seed = st.number_input("Challenge seed", 0, 1000, 7)
rate = st.slider("Learning rate", .01, .40, .25, .01) if case == 0 else .25
z = st.slider("Pre-activation z", -20., 20., 12.) if case == 2 else 12.
lesson = ["optimizers", "ml_practical", "activations", "backprop"][case]
record = prediction_gate(f"debug_{case}", title=cases[case], lesson=lesson, seed=seed,
    settings={"case": cases[case], "learning_rate": rate, "z": z},
    prompt="Predict the cause before running")
if record:
    run_seed = int(record['seed'])
    if case == 0:
        st.code("loss = 5 * w**2\ngradient = 10 * w\nw = w - learning_rate * gradient")
        trace = learning_rate_run(rate)
        st.line_chart(pd.DataFrame({"loss":trace[:,1]}))
        finding = f"rate={rate}; first loss={trace[0,1]:.5g}; final loss={trace[-1,1]:.5g}"
        hint = "The update multiplies w by (1 − 10η). What must its absolute value be for convergence?"
        answer = "The loss has curvature 10. Convergence requires |1−10η| < 1, so 0 < η < 0.2. At η=0.25 the magnitude grows each step."
        lesson = "optimizers"
    elif case == 1:
        leaked, honest, labels = leakage_run(run_seed)
        st.dataframe({"split": ["Repeated entities in train and test", "Unseen entities held out"], "accuracy":[leaked,honest]}, hide_index=True)
        st.write("Each entity has a random binary label and repeated identical observations. The first model memorizes an ID → label lookup. An unseen ID falls back to the training-majority class.")
        finding = f"seed={run_seed}; repeated-ID accuracy={leaked:.3f}; unseen-ID accuracy={honest:.3f}"
        hint = "Does the test set contain an entity whose label was already available in training?"
        answer = "The first split measures memorization of repeated entities. Split by entity before preprocessing. The honest score is near chance because labels were generated randomly."
        lesson = "ml_practical"
    elif case == 2:
        a = 1/(1+np.exp(-z)); derivative=a*(1-a)
        st.metric("Sigmoid derivative", f"{derivative:.8f}")
        finding = f"z={z}; derivative={derivative:.8g}"
        hint = "Compare z=0 with a large positive or negative z. Which term of the chain rule nearly vanishes?"
        answer = "The sigmoid saturates: its derivative approaches zero. Inspect input scale and initialization; normalization or another activation may help, depending on the architecture."
        lesson = "activations"
    else:
        w = 2.; eps=1e-5
        numerical=((w+eps-3)**2-(w-eps-3)**2)/(2*eps)
        wrong=2*(3-w)
        st.code("loss = (w - 3)**2\nimplemented_gradient = 2 * (3 - w)")
        st.dataframe({"check":["implemented","central difference"],"gradient":[wrong,numerical]},hide_index=True)
        finding=f"implemented={wrong}; finite_difference={numerical}"
        hint = "Differentiate the square while retaining the sign of (w − 3)."
        answer = "The derivative is 2(w−3), not 2(3−w). The wrong sign performs ascent. Check central differences before trusting a training curve."
        lesson = "backprop"
    hints(f"debug_{case}", ["Check the simplest baseline and change only one setting.", hint])
    explanation = st.text_area("My diagnosis and proposed fix", max_chars=2000, key=f"debug_explanation_{case}")
    if explanation.strip():
        with st.expander("Compare with the worked diagnosis"):
            st.write(answer)
    save_result(record, finding)
    st.page_link(f"views/{lesson}.py", label="Revisit the underlying concept")
st.page_link("views/my_learning.py", label="Open my notebook and compare runs")
st.caption("Scope: synthetic controlled examples, not evidence about a real dataset. References: the Backprop, Optimizers, Activations and Practical ML lessons.")
