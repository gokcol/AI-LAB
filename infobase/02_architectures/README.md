# Architectures

An architecture encodes how information may interact and where parameters are shared.

- A dense layer mixes features; an MLP alternates mixing with nonlinearities.
- A CNN reuses local filters across positions. Weight sharing is its defining constraint.
- An RNN reuses a transition across sequence steps; the hidden state carries information.
- Attention forms input-dependent weighted mixtures of value vectors. A causal mask
  forbids a token from using future tokens.

The GUI CNN and RNN lessons provide small demonstrations. Standalone training experiments
e14–e17 remain planned. Runnable e18–e20 build attention and a Transformer block in NumPy.
Those weights are seeded and untrained; e21 trains an actual Transformer in PyTorch.

Checkpoint: draw the data flow, annotate shapes, and explain which parameters are reused.
Sources: [Deep Learning, chapters 9–10](https://www.deeplearningbook.org/),
[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).
