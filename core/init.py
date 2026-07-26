"""Weight initialization schemes (Tier 2 e10).

`shape` is (fan_out, fan_in) to match a dense weight matrix. Good init keeps the signal
and gradient variance roughly constant across layers (avoids vanishing/exploding).
"""

from __future__ import annotations

import numpy as np


def zeros(shape):
    return np.zeros(shape)


def normal(shape, std=0.5, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, std, size=shape)


def xavier(shape, rng=None):
    """Glorot: std = sqrt(1 / fan_in). Good for tanh/sigmoid.

    Not a tuned constant. Var(z) = fan_in * Var(w) * Var(x) for z = sum(w_i x_i), so
    demanding Var(z) == Var(x) forces Var(w) = 1/fan_in. Derived in the GUI at
    Math X3 section 10, with the layer-by-layer measurement in its Playground.
    """
    rng = rng or np.random.default_rng()
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=shape)


def he(shape, rng=None):
    """Kaiming: std = sqrt(2 / fan_in). Good for ReLU.

    The 2 is the ReLU factor and nothing else: ReLU zeroes the negative half of a
    symmetric z, so E[relu(z)^2] = E[z^2]/2 and each layer would otherwise halve the
    variance. Doubling Var(w) puts it back. Same derivation, Math X3 section 10.
    """
    rng = rng or np.random.default_rng()
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)
