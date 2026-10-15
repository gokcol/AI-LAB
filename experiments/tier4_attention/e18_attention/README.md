# e18 — attention

Prerequisites: vectors, softmax, and the GUI Attention lesson. About 20 minutes.

**Hypothesis:** causal mixing can add context without letting a position see its future.

Run `python experiments/tier4_attention/e18_attention/run.py` from the repository root after installing the lab.

**Observe:** tensor shapes, normalized attention rows, and the causal mask. The weights
are seeded and untrained. This is a forward pass, not a text generator.

**Rebuild:** write the operation from memory; compare against `core/attention.py`.
**Break:** remove the causal mask and perturb the final input. Explain which earlier
outputs change. Record your seed, prediction, result, and explanation in My learning.

**Check:** rows sum to one, future weights are zero, and residuals preserve shape.
Source: [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
