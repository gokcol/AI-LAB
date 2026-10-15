"""Bounded, deterministic numerical examples for the second set of guided lessons."""
import math


def flag_counts(prior, catch_rate, false_alarm_rate, population=1000):
    """Expected counts, not a random sample. The conditional probabilities are explicit."""
    tp = population * prior * catch_rate
    fn = population * prior * (1-catch_rate)
    fp = population * (1-prior) * false_alarm_rate
    tn = population * (1-prior) * (1-false_alarm_rate)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "posterior": tp/(tp+fp) if tp+fp else None}


def descent(curvature, rate, start, steps, target=0.):
    """Stop before huge values; never label a truncated trajectory converged."""
    w, rows = float(start), []
    for step in range(steps+1):
        loss = curvature*(w-target)**2
        if not math.isfinite(loss) or loss > 1e12:
            return rows, True
        rows.append({"Step": step, "w": w, "Loss": loss})
        w -= rate*2*curvature*(w-target)
    return rows, False


def metrics(tp, fp, fn, tn):
    total = tp+fp+fn+tn
    return {"Accuracy": (tp+tn)/total if total else None,
            "Precision": tp/(tp+fp) if tp+fp else None,
            "Recall": tp/(tp+fn) if tp+fn else None,
            "F1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


# (TP, FP, FN, TN). All candidates see the same 100 rows in each split,
# including 10 positives. Test rows are different from validation rows.
VALIDATION = {"Always negative": (0, 0, 10, 90),
              "Cautious filter": (4, 1, 6, 89),
              "Broad filter": (9, 12, 1, 78)}
TEST = {"Always negative": (0, 0, 10, 90),
        "Cautious filter": (3, 2, 7, 88),
        "Broad filter": (8, 14, 2, 76)}
