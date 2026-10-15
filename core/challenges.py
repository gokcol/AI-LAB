"""Deterministic failure cases with explicit ground truth for learner challenges."""
import numpy as np


def learning_rate_run(rate=.05, steps=24):
    w = 2.0
    trace = []
    for _ in range(steps):
        loss = 5*w*w
        grad = 10*w
        trace.append((w, loss, grad))
        w -= rate*grad
    return np.array(trace)


def leakage_run(seed=7):
    """Random labels per entity: memorizing identity cannot generalize to new entities."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 120)
    # Each entity has an exact repeated observation in the naive holdout.
    leaked_accuracy = float((labels == labels).mean())
    majority = int(np.bincount(labels[:80], minlength=2).argmax())
    honest_accuracy = float((labels[80:] == majority).mean())
    return leaked_accuracy, honest_accuracy, labels


def classification_metrics(y, probabilities, threshold=.5):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(probabilities) >= threshold
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    return {"accuracy": (tp+tn)/len(y), "precision": tp/max(1,tp+fp),
            "recall": tp/max(1,tp+fn), "TP":tp,"FP":fp,"FN":fn,"TN":tn}


FACTS = [
    "The library opens at 09:00 on weekdays.",
    "The library closes at 17:00 on weekdays.",
    "The cafe opens at 08:00 on weekdays.",
    "Library membership is free for students.",
]
QUESTIONS = [
    ("When does the library open on weekdays?", 0),
    ("When does the library close on weekdays?", 1),
    ("Is library membership free for students?", 3),
    ("When does the library open on Sunday?", None),
]


def retrieval_scores(question):
    import re
    def words(s):
        # Deliberately simple lexical retrieval; expose its limits.
        return set(re.findall(r"[a-z]+", s.lower())) - {"the","at","on","is","does","when","for"}
    q = words(question)
    return np.array([len(q & words(fact))/max(1,len(q | words(fact))) for fact in FACTS])
