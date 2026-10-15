"""MNIST bridge: matched initialization, batches and SGD; explicit download only."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from core.digits import mnist, initialize, train_numpy, loss_gradients

p = argparse.ArgumentParser()
p.add_argument('--download', action='store_true')
p.add_argument('--data', default=str(Path(__file__).resolve().parents[3] / 'datasets' / 'mnist'))
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--train', type=int, default=3000)
p.add_argument('--test', type=int, default=500)
p.add_argument('--seed', type=int, default=7)
a = p.parse_args()
if not 1 <= a.epochs <= 100 or not 10 <= a.train <= 60000 or not 10 <= a.test <= 10000:
    p.error('epochs 1..100, train 10..60000, test 10..10000')
try:
    X,y,Xtest,ytest = mnist(a.data, a.download, a.train, a.test)
except FileNotFoundError as exc:
    p.exit(2, str(exc)+'\n')
initial = initialize(seed=a.seed)
with threadpool_limits(limits=1):
    model, history = train_numpy(X,y,initial,epochs=a.epochs,seed=a.seed)
loss, probs, _ = loss_gradients(Xtest,ytest,model)
print('NumPy train cross-entropy:', np.round(history,4))
print('Held-out cross-entropy:', round(loss,4))
print('Held-out accuracy:', round(float((probs.argmax(1)==ytest).mean()),4))
baseline = np.bincount(y, minlength=10).argmax()
print('Training-majority baseline on test:', round(float((ytest==baseline).mean()),4))
from core.digits import train_torch
other, torch_history = train_torch(X,y,initial,epochs=a.epochs,seed=a.seed)
print('PyTorch train cross-entropy:', np.round(torch_history,4))
error = max(np.max(np.abs(model[k]-other[k])) for k in model)
print('Largest final parameter difference:', error)
assert error < 1e-7, 'Framework parity failed; inspect precision and batch order.'
