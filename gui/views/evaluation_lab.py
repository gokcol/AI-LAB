"""Evaluate a misleading classifier and a retrieval system with answerable/unanswerable queries."""
import numpy as np
import pandas as pd
import streamlit as st
from gui.practice import prediction_gate, save_result, hints
from core.challenges import classification_metrics, FACTS, QUESTIONS, retrieval_scores

st.title("Evaluation challenges")
st.caption("Prerequisites: classification and retrieval. Goal: explain what a score measures and when evidence is insufficient. About 20 minutes. Updated 2026-09-20.")
st.markdown("### 1 · A classifier that looks good")
st.write("A dataset has 95 healthy machines and 5 failures. A model predicts healthy for every machine.")
threshold=st.slider("Compare a scored model at this threshold",0.,1.,.5,.05)
record = prediction_gate("evaluation_classifier", title="Accuracy versus recall", lesson="ml_classification",
    seed=7, settings={"healthy":95, "failures":5, "threshold":threshold},
    prompt="Predict the always-healthy model's accuracy and failure recall. Will the scored model find more failures?")
if record:
    y=np.array([0]*95+[1]*5)
    rng=np.random.default_rng(7)
    probabilities=np.r_[rng.uniform(.01,.35,95), rng.uniform(.2,.8,5)]
    a=classification_metrics(y,np.zeros(100),.5)
    b=classification_metrics(y,probabilities,threshold)
    st.dataframe(pd.DataFrame([a,b],index=["always healthy","scored model"]),width="stretch")
    hints("evaluation_classifier", ["How many actual failures are predicted positive?", "Recall divides detected failures by all five actual failures; accuracy counts the 95 easy negatives too."])
    st.info("Choose a threshold using validation data and the cost of false alarms versus missed failures. These scores are synthetic, not a deployment recommendation.")
    save_result(record, f"always healthy={a}; scored model={b}")
st.markdown("### 2 · Retrieval is not proof")
st.write("This tiny corpus has weekday information. Inspect the evidence before accepting an answer.")
for i,fact in enumerate(FACTS):
    st.write(f"Document {i+1}: {fact}")
qi=st.selectbox("Question",range(len(QUESTIONS)),format_func=lambda i: QUESTIONS[i][0])
question,gold=QUESTIONS[qi]
scores=retrieval_scores(question)
k=st.slider("Documents retrieved (k)",1,4,1)
retrieval_record = prediction_gate("evaluation_retrieval", title="Retrieval versus evidence", lesson="embeddings",
    seed="deterministic", settings={"question":question,"k":k},
    prompt="Predict whether the retrieved documents will support the exact question. What evidence would be missing?")
if retrieval_record:
    rank=np.argsort(-scores,kind='stable')[:k]
    st.dataframe(pd.DataFrame({"document":rank+1,"similarity":scores[rank].round(3),"evidence":[FACTS[i] for i in rank]}),hide_index=True)
    verdict=st.radio("Does the retrieved text support an answer to this exact question?",["Choose…","Supported","Insufficient evidence"],key=f"eval_verdict_{qi}")
    if verdict != "Choose…":
        supported=gold is not None and gold in rank
        correct=verdict == ("Supported" if supported else "Insufficient evidence")
        (st.success if correct else st.warning)("Correct." if correct else "Inspect the day and subject in the retrieved passage.")
        if gold is None:
            st.write("Insufficient evidence: weekday opening hours do not establish Sunday hours. A high similarity score cannot fill that gap.")
        elif supported:
            st.write(f"Supporting evidence: document {gold+1}. Only answer what this passage establishes.")
        else:
            st.write(f"Retrieval miss: document {gold+1} contains the answer, but it was not retrieved. Increase k and inspect again.")
    hits=sum(g is not None and g in np.argsort(-retrieval_scores(q),kind='stable')[:k] for q,g in QUESTIONS)
    st.metric("Recall@k on the three answerable questions", f"{hits}/3")
    st.caption("The unanswerable Sunday question is excluded from retrieval recall and evaluated separately for abstention. This is lexical retrieval and a human evidence check; no LLM is called or scored.")
    hints("evaluation_retrieval", ["Match the subject and day, not just overlapping words.", "A relevant-looking passage can miss the answer; no amount of weekday evidence establishes Sunday hours."])
    save_result(retrieval_record, f"retrieved documents={(rank+1).tolist()}; support={gold is not None and gold in rank}; answerable recall={hits}/3")
st.page_link("views/ml_classification.py",label="Continue: classification thresholds")
st.page_link("views/embeddings.py",label="Continue: RAG evaluation and failure modes")
