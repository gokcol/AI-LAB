"""Scientific checks for the learner-trackable training simulations."""

import numpy as np
import pytest

from core import training as t
from gui import i18n
from gui.catalog_tr import TR


def _run(step_fn, params, steps):
    losses = []
    for epoch in range(steps):
        step = step_fn(params, epoch)
        losses.append(step.loss)
        params = step.parameters_after
    return params, np.asarray(losses)


def test_delivery_is_deterministic_and_gradient_matches_finite_difference():
    data = t.delivery_data(n=18, noise=1.2, seed=11)
    same = t.delivery_data(n=18, noise=1.2, seed=11)
    assert np.array_equal(data.X_train, same.X_train)
    p = np.array([0.4, -0.2])
    step = t.regression_step(data.X_train, data.y_train, p)
    numeric = t.finite_difference(
        lambda q: t.regression_values(data.X_train, data.y_train, q)[2], p
    )
    assert step.gradients == pytest.approx(numeric, abs=1e-5)
    assert step.parameters_after == pytest.approx(p + step.delta)
    assert step.metrics["batch_mse"] == pytest.approx(step.loss)
    assert step.verification["update_max_error"] == 0.0


def test_delivery_training_approaches_ols_without_touching_test_rows():
    data = t.delivery_data(n=48, noise=1.0, seed=4)
    p, losses = _run(lambda q, e: t.regression_step(data.X_train, data.y_train, q, lr=.004, epoch=e),
                     np.zeros(2), 5000)
    X = np.c_[data.X_train[:, 0], np.ones(len(data.X_train))]
    ols, *_ = np.linalg.lstsq(X, data.y_train, rcond=None)
    assert losses[-1] < losses[0]
    assert p == pytest.approx(ols, abs=.08)
    assert set(data.train_indices).isdisjoint(data.test_indices)


def test_delivery_unsafe_learning_rate_visibly_diverges():
    data = t.delivery_data(n=32, noise=2.0, seed=4)
    p = np.zeros(2)
    for _ in range(50):
        step = t.regression_step(data.X_train, data.y_train, p, lr=.006)
        p = step.parameters_after
        if step.loss > 1e10:
            break
    assert step.loss > 1e10


def test_logistic_gradient_and_split_integrity():
    data = t.machine_data(n=100, seed=5)
    p = np.array([.1, -.2, .05])
    step = t.logistic_step(data.X_train, data.y_train, p)
    numeric = t.finite_difference(
        lambda q: t.logistic_values(data.X_train, data.y_train, q)[3], p
    )
    assert step.gradients == pytest.approx(numeric, abs=1e-5)
    assert set(data.train_indices).isdisjoint(data.test_indices)
    # Scaling was fitted only on the training rows, so their feature mean is zero.
    assert data.X_train.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_logistic_default_training_beats_majority_baseline():
    data = t.machine_data(n=120, seed=4)
    p, losses = _run(lambda q, e: t.logistic_step(data.X_train, data.y_train, q, lr=.15, epoch=e),
                     np.zeros(3), 160)
    _, probabilities, _, _, _ = t.logistic_values(data.X_test, data.y_test, p)
    accuracy = ((probabilities >= .5) == data.y_test).mean()
    majority = max(data.y_test.mean(), 1 - data.y_test.mean())
    assert losses[-1] < losses[0]
    assert accuracy > majority


def test_xor_gradients_and_default_training():
    p = t.xor_init(1)
    step = t.xor_step(p)
    numeric = t.finite_difference(lambda q: t.xor_values(t.XOR_X, t.XOR_Y, q)["loss"], p)
    assert step.gradients == pytest.approx(numeric, abs=1e-5)
    p, losses = _run(lambda q, e: t.xor_step(q, lr=.1, epoch=e), p, 900)
    pred = t.xor_values(t.XOR_X, t.XOR_Y, p)["predictions"]
    assert losses[-1] < losses[0]
    assert ((pred >= 0) == (t.XOR_Y >= 0)).sum() == 4


def test_checkpoint_replay_is_exact():
    p = t.xor_init(3)
    first = t.xor_step(p, lr=.1, epoch=0)
    replay = t.xor_step(first.parameters_before, lr=.1, epoch=0)
    assert replay.parameters_after == pytest.approx(first.parameters_after)
    second = t.xor_step(first.parameters_after, lr=.1, epoch=1)
    assert second.parameters_before == pytest.approx(first.parameters_after)


def test_training_controls_have_active_turkish_ui_coverage():
    assert i18n.MULTILINGUAL is True
    for label in ("🏋 Train step by step", "🔬 Training microscope", "One training step",
                  "🏋 Train next-token model"):
        assert label in TR


def test_bigram_invariants_and_training_beats_uniform_baseline():
    vocab, train, valid = t.text_pairs()
    weights = np.zeros((len(vocab), len(vocab)))
    step = t.bigram_step(weights, train[:, 0], train[:, 1])
    probs = step.predictions
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(probs)))
    assert step.loss == pytest.approx(np.log(len(vocab)))
    assert step.gradients.sum(axis=1) == pytest.approx(np.zeros(len(vocab)), abs=1e-12)
    numeric = t.finite_difference(
        lambda q: t.bigram_values(q, train[:, 0], train[:, 1])[3], weights
    )
    assert step.gradients == pytest.approx(numeric, abs=1e-5)
    single = t.bigram_step(weights, train[:1, 0], train[:1, 1], lr=.5)
    target = train[0, 1]
    assert single.parameters_after[train[0, 0], target] > 0
    weights, losses = _run(
        lambda q, e: t.bigram_step(q, train[:, 0], train[:, 1], lr=.8, epoch=e), weights, 120
    )
    _, _, _, valid_loss, _ = t.bigram_values(weights, valid[:, 0], valid[:, 1])
    assert losses[-1] < losses[0]
    assert valid_loss < np.log(len(vocab))
