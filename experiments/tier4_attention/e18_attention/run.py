"""Seeded NumPy forward-pass experiment; no pretrained model or downloads."""
import numpy as np
X = np.random.default_rng(7).normal(size=(4,8))
from core.attention import attention, projections
w = projections(8, 7)
out, weights = attention(X @ w['q'], X @ w['k'], X @ w['v'])
assert np.allclose(weights.sum(axis=1), 1)
assert np.allclose(np.triu(weights, 1), 0)
print("Masked attention weights:\n", weights.round(3))
print("Output shape:", out.shape)
