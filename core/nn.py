"""A tiny neural-net library built on the autograd engine (Tier 1, e05).

Neuron / Layer / MLP of `core.engine.Value` scalars — the trainable counterpart to the
NumPy forward-only `core.neuron`. This is the same shape as Karpathy's `micrograd.nn`.
Train it with `core.optim` against a `Value` loss; `loss.backward()` fills every
parameter's `.grad`.
"""

from __future__ import annotations

import random

from core.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin="tanh", rng=None):
        rng = rng or random
        # NOTE: uniform(-1, 1) gives Var(w) = 1/3 regardless of fan_in, which is NOT the
        # Xavier/He scaling derived in core/init.py and taught in Math X3 section 10. It is
        # kept because these demos are two layers deep, train in seconds, and several tuned
        # seeds in the GUI depend on it; at this depth the difference is cosmetic. It does
        # mean an untrained MLP starts at a cross-entropy above ln(C) rather than at it --
        # see the debugging ladder on the MLP page, which uses this as its worked example.
        # For anything deep, use core.init.xavier / core.init.he.
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # w·x + b
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        if self.nonlin == "sigmoid":
            return act.sigmoid()
        return act  # linear

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Stack of layers. `nouts` are the layer widths; hidden layers use `nonlin`,
    the output layer uses `out_nonlin` (default linear)."""

    def __init__(self, n_in, nouts, nonlin="tanh", out_nonlin=None, seed=None):
        if seed is not None:
            random.seed(seed)
        sizes = [n_in] + list(nouts)
        self.layers = []
        for i in range(len(nouts)):
            last = i == len(nouts) - 1
            self.layers.append(
                Layer(sizes[i], sizes[i + 1], nonlin=(out_nonlin if last else nonlin))
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
