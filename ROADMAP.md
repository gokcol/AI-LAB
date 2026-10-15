# Roadmap

Each tier adds new pieces to `core/` and **reuses everything before it**. Check
items off as you go. Status legend: `[ ]` todo · `[~]` in progress · `[x]` done.

---

## Tier 0 — The Neuron  *(pure NumPy, by hand)*
- [x] **e01 — Single neuron**: forward pass, weights, bias, activation. *(worked reference)*
- [x] **e02 — Perceptron gates**: the perceptron learning rule; trains AND / OR (converges).
- [x] **e03 — XOR fails**: a single linear neuron cannot learn XOR → motivates layers.

**Adds to core/:** `activations.py`, `neuron.py` (NumPy forward), `viz.py`.

## Tier 1 — Backprop & the autograd engine
- [x] **e04 — Autograd engine**: `core/engine.py` `Value` (scalar reverse-mode autodiff).
- [x] **e05 — MLP solves XOR**: a `Value`-based MLP (`core/nn.py`) trains to 4/4 on XOR.
- [x] **e06 — Manual backprop**: gradient-check vs central differences (error ~1e-11).
- [x] **e07 — Activation zoo**: sigmoid vs ReLU — the vanishing-gradient demo.
- [x] **e08 — Loss functions**: MSE vs cross-entropy; why CE for classification.

**Adds to core/:** `engine.py` (Value), `losses.py`, `optim.py`, `init.py`.

## Tier 2 — Training dynamics  *(vectorized NumPy → bridge to PyTorch)*
- [x] e09 — Optimizers: SGD → Momentum → Adam on the same loss surface.
- [x] e10 — Initialization (Xavier/He) & signal-scale effects.
- [x] e11 — Regularization: L2, over/underfitting.
- [x] e12 — **MNIST from scratch** (pure-NumPy MLP).
- [x] e13 — Reproduce e12 in **PyTorch**; compare.

## Tier 3 — Architectures  *(PyTorch)*
- [ ] e14 — CNN (LeNet) on MNIST.
- [ ] e15 — RNN/LSTM char-level model.
- [ ] e16 — Embeddings (mini word2vec).
- [ ] e17 — Seq2seq + the attention bottleneck.

## Tier 4 — Attention & Transformers
- [x] e18 — Attention from scratch (single head, by hand).
- [x] e19 — Multi-head self-attention.
- [x] e20 — A full Transformer block (residual + LayerNorm + MLP).
- [x] e21 — **nanoGPT-style char-level GPT** on a tiny corpus.
- [ ] e22 — BPE tokenization.

## Tier 5 — LLM
- [ ] e23 — Sampling: greedy / temperature / top-k / top-p.
- [ ] e24 — Fine-tune a small pretrained model.
- [ ] e25 — LoRA / parameter-efficient fine-tuning.
- [ ] e26 — Instruction tuning & RLHF/DPO (read + small demo).
- [ ] e27 — Embeddings + RAG; inference & KV-cache basics.

---

### The two "aha" milestones to aim for
1. **End of Tier 1:** you wrote an autograd engine and an MLP that learns XOR — you
   now understand *exactly* what PyTorch does under the hood.
2. **End of Tier 4:** you trained a tiny GPT that generates text — you now understand
   *exactly* what an LLM is.
