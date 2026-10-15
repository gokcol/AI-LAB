"""A vectorized NumPy classifier and identical PyTorch bridge.

MNIST files use the original IDX format. Downloads occur only by explicit request.
"""
from pathlib import Path
import gzip
import struct
import urllib.request
import numpy as np
from core.training import softmax

MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


def read_idx(path):
    with gzip.open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise ValueError("Truncated IDX file.")
        zero, kind, dims = struct.unpack(">HBB", header)
        if zero or kind != 8 or dims not in (1, 3):
            raise ValueError("Expected unsigned-byte MNIST IDX data.")
        shape = struct.unpack(">" + "I" * dims, f.read(4 * dims))
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        if np.prod(shape) != raw.size:
            raise ValueError("IDX shape does not match payload.")
        return raw.reshape(shape)


def mnist(directory, download=False, n_train=3000, n_test=500):
    directory = Path(directory)
    for name in FILES:
        target = directory / name
        if not target.exists():
            if not download:
                raise FileNotFoundError("MNIST is not cached. Run this experiment locally with --download.")
            directory.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(MNIST_BASE + name, timeout=30) as response:
                blob = response.read(20_000_001)
            if len(blob) > 20_000_000:
                raise ValueError("Unexpectedly large MNIST download.")
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(blob)
            read_idx(temporary)  # validate before publishing the cache file
            temporary.replace(target)
    Xtr,ytr,Xte,yte = [read_idx(directory / name) for name in FILES]
    return (Xtr[:n_train].reshape(-1, 784).astype(float)/255, ytr[:n_train].astype(int),
            Xte[:n_test].reshape(-1, 784).astype(float)/255, yte[:n_test].astype(int))


def initialize(n_input=784, hidden=32, classes=10, seed=7):
    rng = np.random.default_rng(seed)
    return {"w1": rng.normal(0, np.sqrt(2/n_input), (n_input, hidden)), "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(1/hidden), (hidden, classes)), "b2": np.zeros(classes)}


def loss_gradients(X, y, p):
    z = X @ p["w1"] + p["b1"]
    h = np.maximum(z, 0)
    logits = h @ p["w2"] + p["b2"]
    probs = softmax(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y])
    dz2 = probs.copy(); dz2[np.arange(len(y)), y] -= 1; dz2 /= len(y)
    dz1 = (dz2 @ p["w2"].T) * (z > 0)
    grads = {"w1": X.T @ dz1, "b1": dz1.sum(axis=0), "w2": h.T @ dz2, "b2": dz2.sum(axis=0)}
    return float(loss), probs, grads


def train_numpy(X, y, parameters, epochs=8, lr=.15, batch=128, seed=7):
    p = {k:v.copy() for k,v in parameters.items()}
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        indices = rng.permutation(len(y))
        for start in range(0, len(y), batch):
            ix = indices[start:start+batch]
            _, _, grad = loss_gradients(X[ix], y[ix], p)
            for k in p:
                p[k] -= lr * grad[k]
        history.append(loss_gradients(X, y, p)[0])
    return p, history


def train_torch(X, y, parameters, epochs=8, lr=.15, batch=128, seed=7):
    import torch
    torch.set_num_threads(1)
    p = {k:torch.tensor(v.copy(), dtype=torch.float64, requires_grad=True) for k,v in parameters.items()}
    inputs, labels = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    history = []
    def logits(ix):
        return torch.relu(inputs[ix] @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
    for _ in range(epochs):
        indices = rng.permutation(len(y))
        for start in range(0,len(y),batch):
            ix = indices[start:start+batch]
            loss = torch.nn.functional.cross_entropy(logits(ix), labels[ix])
            loss.backward()
            with torch.no_grad():
                for value in p.values():
                    value -= lr * value.grad
                    value.grad.zero_()
        with torch.no_grad():
            history.append(float(torch.nn.functional.cross_entropy(logits(slice(None)), labels)))
    return {k:v.detach().numpy() for k,v in p.items()}, history
