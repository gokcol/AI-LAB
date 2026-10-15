"""Seeded NumPy forward-pass experiment; no pretrained model or downloads."""
import numpy as np
X = np.random.default_rng(7).normal(size=(4,8))
from core.attention import transformer_block, block_weights
w = block_weights(8, 7)
out, trace = transformer_block(X, w)
future = X.copy(); future[-1] += 10
changed, _ = transformer_block(future, w)
error = np.max(np.abs(changed[:-1] - out[:-1]))
assert error < 1e-12
print("Input -> attention residual -> block:", X.shape, trace['residual'].shape, out.shape)
print("Future-token leakage:", error)
