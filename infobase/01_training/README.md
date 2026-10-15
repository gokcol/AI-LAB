# Training

Training repeats four operations: forward pass, loss, gradient, update. Evaluate on
held-out data to distinguish fitting from generalization.

- **Optimizer:** chooses the update. Compare SGD, momentum and Adam in e09.
- **Initialization:** controls signal scale before learning. Inspect e10.
- **Regularization:** constrains the fit; choose strength on validation, not test data (e11).
- **Framework bridge:** e12 implements a NumPy MLP on MNIST; e13 repeats identical SGD
  steps with PyTorch autograd. Matching batches and weights makes this a useful check.

Checkpoint: derive one weight gradient, verify it by central differences, and explain
why test accuracy can fall while training loss decreases.

See `experiments/tier2_training/`, the GUI Optimizers and Regularization lessons,
and [Goodfellow et al., Deep Learning, chapters 7–8](https://www.deeplearningbook.org/).
