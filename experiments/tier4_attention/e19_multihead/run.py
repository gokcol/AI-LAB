"""Seeded NumPy forward-pass experiment; no pretrained model or downloads."""
import numpy as np
X = np.random.default_rng(7).normal(size=(4,8))
from core.attention import multihead, projections
for heads in (1, 2, 4, 8):
    out, maps = multihead(X, projections(8, 7), heads)
    assert out.shape == X.shape
    assert np.allclose(maps.sum(axis=-1), 1)
    print(f"{heads} heads: maps {maps.shape}, output {out.shape}")
