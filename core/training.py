"""Deterministic, inspectable training loops used by the teaching simulations.

The functions here deliberately return the intermediate values a learner needs to
audit a step: predictions, per-example loss terms, gradients, and parameter
updates.  They contain no Streamlit code, so the numerical claims can be tested
independently of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class Split:
    """A deterministic train/test split, with optional raw feature values."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    raw_train: np.ndarray | None = None
    raw_test: np.ndarray | None = None


@dataclass(frozen=True)
class Step:
    """One SGD-style update, retained for replay and calculation ledgers."""

    epoch: int
    batch_indices: np.ndarray
    parameters_before: np.ndarray
    predictions: np.ndarray
    per_example_loss: np.ndarray
    loss: float
    gradients: np.ndarray
    delta: np.ndarray
    parameters_after: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    verification: dict[str, float] = field(default_factory=dict)


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _split_indices(n: int, test_fraction: float, seed: int, y=None):
    """Split indices deterministically; stratify binary labels when supplied."""
    rng = np.random.default_rng(seed + 10_003)
    if y is None:
        idx = rng.permutation(n)
        cut = max(1, min(n - 1, int(round(n * (1 - test_fraction)))))
        return np.sort(idx[:cut]), np.sort(idx[cut:])
    y = np.asarray(y)
    train, test = [], []
    for cls in np.unique(y):
        group = rng.permutation(np.flatnonzero(y == cls))
        cut = max(1, min(len(group) - 1, int(round(len(group) * (1 - test_fraction)))))
        train.extend(group[:cut]); test.extend(group[cut:])
    return np.sort(np.asarray(train)), np.sort(np.asarray(test))


def delivery_data(n=32, noise=2.0, seed=0, test_fraction=0.25) -> Split:
    """Fake but meaningful delivery-distance → delivery-time regression data."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1.0, 20.0, int(n))
    minutes = 8.0 + 2.2 * distance + rng.normal(0.0, float(noise), int(n))
    train, test = _split_indices(len(distance), test_fraction, seed)
    return Split(distance[train, None], minutes[train], distance[test, None], minutes[test],
                 train, test, distance[train, None], distance[test, None])


def machine_data(n=96, seed=0, test_fraction=0.25) -> Split:
    """Simulated machine sensors: temperature + vibration → failure probability.

    The labels are sampled from a known logistic process.  That keeps the story
    meaningful without making it perfectly separable.
    """
    rng = np.random.default_rng(seed)
    raw = np.column_stack((rng.normal(70.0, 8.0, int(n)),
                           rng.normal(0.60, 0.20, int(n))))
    latent = -0.9 + 1.0 * ((raw[:, 0] - 70.0) / 8.0) + 1.35 * ((raw[:, 1] - .60) / .20)
    y = rng.binomial(1, sigmoid(latent)).astype(float)
    train, test = _split_indices(len(raw), test_fraction, seed, y)
    raw_train, raw_test = raw[train], raw[test]
    mu, sd = raw_train.mean(axis=0), raw_train.std(axis=0) + EPS
    return Split((raw_train - mu) / sd, y[train], (raw_test - mu) / sd, y[test],
                 train, test, raw_train, raw_test)


def regression_values(X, y, parameters):
    """Return predictions, individual MSE terms, scalar loss and exact gradient."""
    X = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)
    w, b = np.asarray(parameters, dtype=float)
    pred = w * X + b
    residual = pred - y
    terms = residual ** 2
    grad = np.array([2.0 * np.mean(X * residual), 2.0 * np.mean(residual)])
    return pred, terms, float(terms.mean()), grad


def regression_step(X, y, parameters, lr=0.03, epoch=0, batch_indices=None) -> Step:
    before = np.asarray(parameters, dtype=float).copy()
    pred, terms, loss, grad = regression_values(X, y, before)
    delta = -float(lr) * grad
    return Step(epoch, np.arange(len(pred)) if batch_indices is None else np.asarray(batch_indices),
                before, pred, terms, loss, grad, delta, before + delta,
                {"batch_mse": loss}, {"update_max_error": 0.0})


def logistic_values(X, y, parameters):
    """Binary logistic regression with BCE and its simplified exact gradient."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(parameters, dtype=float)
    logits = X @ p[:-1] + p[-1]
    prob = sigmoid(logits)
    terms = -(y * np.log(prob + EPS) + (1.0 - y) * np.log(1.0 - prob + EPS))
    residual = prob - y
    grad = np.r_[(X.T @ residual) / len(X), residual.mean()]
    return logits, prob, terms, float(terms.mean()), grad


def logistic_step(X, y, parameters, lr=0.15, epoch=0, batch_indices=None) -> Step:
    before = np.asarray(parameters, dtype=float).copy()
    _, prob, terms, loss, grad = logistic_values(X, y, before)
    delta = -float(lr) * grad
    return Step(epoch, np.arange(len(prob)) if batch_indices is None else np.asarray(batch_indices),
                before, prob, terms, loss, grad, delta, before + delta,
                {"batch_bce": loss}, {"update_max_error": 0.0})


XOR_X = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = np.asarray([-1.0, 1.0, 1.0, -1.0])
XOR_PARAMETER_NAMES = ("h1.w1", "h1.w2", "h1.b", "h2.w1", "h2.w2", "h2.b",
                       "out.w1", "out.w2", "out.b")


def xor_init(seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.8, 0.8, 9)


def xor_values(X, y, parameters):
    """A fully spelled-out 2→2→1 tanh MLP forward/backward pass."""
    p = np.asarray(parameters, dtype=float)
    W1 = np.array([[p[0], p[1]], [p[3], p[4]]])
    b1 = np.array([p[2], p[5]])
    W2 = np.array([p[6], p[7]])
    b2 = p[8]
    z1 = X @ W1.T + b1
    h = np.tanh(z1)
    z2 = h @ W2 + b2
    pred = np.tanh(z2)
    residual = pred - y
    terms = residual ** 2
    # Mean squared error, then reverse the same graph explicitly.
    dz2 = (2.0 / len(X)) * residual * (1.0 - pred ** 2)
    dW2 = h.T @ dz2
    db2 = dz2.sum()
    dh = np.outer(dz2, W2)
    dz1 = dh * (1.0 - h ** 2)
    dW1 = dz1.T @ X
    db1 = dz1.sum(axis=0)
    grad = np.array([dW1[0, 0], dW1[0, 1], db1[0], dW1[1, 0], dW1[1, 1], db1[1],
                     dW2[0], dW2[1], db2])
    return {"z1": z1, "hidden": h, "z2": z2, "predictions": pred,
            "terms": terms, "loss": float(terms.mean()), "gradients": grad}


def xor_step(parameters, lr=0.1, epoch=0) -> Step:
    before = np.asarray(parameters, dtype=float).copy()
    values = xor_values(XOR_X, XOR_Y, before)
    delta = -float(lr) * values["gradients"]
    return Step(epoch, np.arange(len(XOR_X)), before, values["predictions"], values["terms"],
                values["loss"], values["gradients"], delta, before + delta,
                {"batch_mse": values["loss"]}, {"update_max_error": 0.0})


TEXT_CORPUS = "the cat sat . the dog ran . the cat ran . the dog sat ."


def text_pairs(corpus=TEXT_CORPUS, validation_fraction=0.2):
    """Chronological bigram pairs and a small, readable shared vocabulary."""
    chars = list(corpus.lower())
    vocab = tuple(sorted(set(chars)))
    to_id = {ch: i for i, ch in enumerate(vocab)}
    pairs = np.asarray([(to_id[a], to_id[b]) for a, b in zip(chars[:-1], chars[1:])], dtype=int)
    cut = max(1, min(len(pairs) - 1, int(round(len(pairs) * (1.0 - validation_fraction)))))
    return vocab, pairs[:cut], pairs[cut:]


def bigram_values(weights, contexts, targets):
    logits = np.asarray(weights, dtype=float)[np.asarray(contexts, dtype=int)]
    probs = softmax(logits)
    targets = np.asarray(targets, dtype=int)
    terms = -np.log(probs[np.arange(len(targets)), targets] + EPS)
    grad_rows = probs.copy()
    grad_rows[np.arange(len(targets)), targets] -= 1.0
    grad = np.zeros_like(weights, dtype=float)
    np.add.at(grad, contexts, grad_rows / len(contexts))
    return logits, probs, terms, float(terms.mean()), grad


def bigram_step(weights, contexts, targets, lr=0.5, epoch=0, batch_indices=None) -> Step:
    before = np.asarray(weights, dtype=float).copy()
    _, probs, terms, loss, grad = bigram_values(before, contexts, targets)
    delta = -float(lr) * grad
    indices = np.arange(len(contexts)) if batch_indices is None else np.asarray(batch_indices)
    return Step(epoch, indices, before, probs, terms, loss, grad, delta, before + delta,
                {"batch_cross_entropy": loss, "batch_perplexity": float(np.exp(loss))},
                {"update_max_error": 0.0, "gradient_row_sum_max": float(np.abs(grad.sum(axis=1)).max())})


def finite_difference(loss_fn, parameters, eps=1e-6):
    """Central-difference gradient used by tests and the learner verification panel."""
    p = np.asarray(parameters, dtype=float)
    out = np.zeros_like(p)
    for i in range(p.size):
        plus, minus = p.copy(), p.copy()
        plus.flat[i] += eps; minus.flat[i] -= eps
        out.flat[i] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * eps)
    return out
