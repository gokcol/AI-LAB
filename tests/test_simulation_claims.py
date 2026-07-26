"""Regression tests for numerical claims made by the interactive simulations.

Render tests prove that pages execute. These tests protect the scientific relationships
that the prose asks learners to observe.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
from scipy.stats import wasserstein_distance

from core.engine import Value
from core.optim import Adam, Momentum, SGD


ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWS = ROOT / "gui" / "views"


def _load_functions(filename: str, names: set[str]):
    """Load pure numeric helpers from a Streamlit page without executing the page."""
    path = VIEWS / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _optimizer_loss(kind, lr=0.2, steps=100, curvature=6.0):
    x, y = Value(-4.5), Value(-3.5)
    if kind == "SGD":
        opt = SGD([x, y], lr=lr)
    elif kind == "Momentum":
        opt = Momentum([x, y], lr=lr, mu=0.9)
    else:
        opt = Adam([x, y], lr=lr)
    for _ in range(steps):
        loss = curvature * x ** 2 + y ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if abs(x.data) > 50 or abs(y.data) > 50:
            return float("inf")
    return curvature * x.data ** 2 + y.data ** 2


def test_optimizer_default_race_matches_page_explanation():
    losses = {name: _optimizer_loss(name) for name in ("SGD", "Momentum", "Adam")}
    assert np.isinf(losses["SGD"])
    assert losses["Adam"] < losses["Momentum"]


def test_rnn_gradient_depends_on_weight_times_tanh_slope():
    def surviving_gradient(weight, lag=30):
        gradient, state = 1.0, 0.3
        for _ in range(lag):
            z = weight * state
            gradient *= weight * (1 - np.tanh(z) ** 2)
            state = np.tanh(z)
        return abs(gradient)

    assert surviving_gradient(1.0) > 0.1
    assert surviving_gradient(1.5) < 1e-9


def test_gelu_has_an_isolated_zero_derivative():
    def gelu(z):
        return 0.5 * z * (
            1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))
        )

    z = np.linspace(-0.8, -0.7, 10_001)
    derivative = (gelu(z + 1e-5) - gelu(z - 1e-5)) / 2e-5
    assert derivative.min() < 0 < derivative.max()


def test_entropy_endpoints_are_exact():
    ns = _load_functions("math_information.py", {"_H"})
    assert ns["_H"](0.0) == 0.0
    assert ns["_H"](1.0) == 0.0
    assert ns["_H"](0.5) == 1.0


def test_diffusion_more_steps_improves_default_sample():
    ns = _load_functions("generative.py", {"_diffusion_samples"})
    sample2 = ns["_diffusion_samples"](2, 1500)
    sample60 = ns["_diffusion_samples"](60, 1500)
    rng = np.random.default_rng(0)
    pick = rng.random(4000) < 0.35
    target = np.where(
        pick, rng.normal(-1.6, 0.35, 4000), rng.normal(1.1, 0.55, 4000)
    )
    assert wasserstein_distance(target, sample60) < wasserstein_distance(target, sample2)
    assert abs(np.std(sample60) - np.std(target)) < abs(np.std(sample2) - np.std(target))


def test_kmeans_restarts_avoid_a_nonmonotonic_demo_curve():
    ns = _load_functions("ml_unsupervised.py", {"_kmeans_once", "_kmeans"})
    kmeans = ns["_kmeans"]
    for seed in (13, 152, 199):
        rng = np.random.default_rng(seed)
        centers = rng.uniform(-4, 4, (3, 2))
        X = np.vstack([rng.normal(c, 0.8, (50, 2)) for c in centers])
        inertia = [kmeans(X, k, seed)[2] for k in range(1, 7)]
        assert np.all(np.diff(inertia) <= 1e-9), (seed, inertia)


def test_regularization_sweep_scores_validation_not_test():
    source = (VIEWS / "regularization.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sweep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sweep")
    body = ast.get_source_segment(source, sweep)
    assert "score(Xval, yval)" in body
    assert "score(Xte, yte)" not in body


def test_stale_fixed_measurements_and_model_identity_do_not_return():
    generative = (VIEWS / "generative.py").read_text(encoding="utf-8")
    deep = (VIEWS / "deep_playground.py").read_text(encoding="utf-8")
    transformer = (VIEWS / "transformer.py").read_text(encoding="utf-8")
    assert "about 18× worse" not in generative
    assert "trough is only 50 % deep" not in generative
    assert "| 5 | 20 | 38 | **40** | 50 | 120 | 2000 |" not in deep
    assert "not a Transformer" in transformer
    assert "Scale this up and it's a GPT" not in transformer
