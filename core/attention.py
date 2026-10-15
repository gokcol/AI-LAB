"""Transparent NumPy attention building blocks, shared by experiments and the GUI."""
import numpy as np
from core.training import softmax


def attention(q, k, v, causal=True):
    q, k, v = (np.asarray(x, dtype=float) for x in (q, k, v))
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Expected Q[Tq,D], K[Tk,D], V[Tk,Dv].")
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        if len(q) != len(k):
            raise ValueError("This causal self-attention demo requires equal sequence lengths.")
        scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), 1), -np.inf, scores)
    weights = softmax(scores)
    return weights @ v, weights


def projections(width=8, seed=7):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(0, 1 / np.sqrt(width), (width, width)) for name in ("q", "k", "v", "out")}


def multihead(x, weights, heads=2, causal=True):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or heads < 1 or x.shape[1] % heads:
        raise ValueError("The embedding width must be divisible by the number of heads.")
    width = x.shape[1] // heads
    q,k,v = (x @ weights[name] for name in ("q", "k", "v"))
    outputs, maps = [], []
    for h in range(heads):
        sl = slice(h * width, (h + 1) * width)
        out, amap = attention(q[:, sl], k[:, sl], v[:, sl], causal)
        outputs.append(out); maps.append(amap)
    return np.concatenate(outputs, axis=-1) @ weights["out"], np.stack(maps)


def layernorm(x, eps=1e-5):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def transformer_block(x, weights, heads=2):
    """Pre-norm residual block; fixed seeded weights, no dropout or training here."""
    att, maps = multihead(layernorm(x), weights, heads)
    residual = x + att
    # A width -> 2*width -> width ReLU feed-forward layer.
    y = residual + np.maximum(0, layernorm(residual) @ weights["ff1"]) @ weights["ff2"]
    return y, {"attention": att, "residual": residual, "maps": maps}


def block_weights(width=8, seed=7):
    weights = projections(width, seed)
    rng = np.random.default_rng(seed + 1)
    weights["ff1"] = rng.normal(0, 1 / np.sqrt(width), (width, 2 * width))
    weights["ff2"] = rng.normal(0, 1 / np.sqrt(2 * width), (2 * width, width))
    return weights
