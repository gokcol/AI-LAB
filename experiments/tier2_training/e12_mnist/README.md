# e12 — MNIST with a NumPy MLP

**Goal:** train a vectorized handwritten-digit classifier with manual backpropagation.
Prerequisites: backpropagation, cross-entropy, minibatches. About 30 minutes.

```bash
pip install -e '.[gui]'
# No PyTorch required for e12
python experiments/tier2_training/e12_mnist/run.py --download
```

Downloads the four original MNIST IDX files once (about 12 MB compressed). To use
existing files, pass `--data /path/to/mnist`. No download occurs without `--download`.
The default uses 3,000 training and 500 test images for a short CPU run; these are
teaching results, not full-dataset benchmarks. Increase `--train 60000 --test 10000`
for the full fixed split. Never tune on the reported test score; introduce a validation
split first if you compare hyperparameters.

**Hypothesis:** a 784 → 32 → 10 network can beat the training-majority baseline on
unseen digits. e13 should reproduce e12's parameters to numerical tolerance when
initialization, batches, precision and SGD are identical.

**Observe:** loss falls; final test accuracy exceeds the baseline. Record seed and
settings. **Break:** raise the learning rate or permute training labels. Explain the result.
**Rebuild:** derive `w2`'s gradient; check against finite differences before training.

Implementation: `core/digits.py`. Sources: [MNIST](https://yann.lecun.com/exdb/mnist/)
and [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
