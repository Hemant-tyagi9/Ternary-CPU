import numpy as np
from pathlib import Path
from .nn_loader import NeuroGate
import os

def one_hot_trit(t):
    return np.array([1,0,0]) if t==-1 else (np.array([0,1,0]) if t==0 else np.array([0,0,1]))

def prepare_dataset(op):
    # op: 'AND','OR','ADD'
    inputs, targets = [], []
    for a in [-1,0,1]:
        for b in [-1,0,1]:
            x = np.concatenate([one_hot_trit(a), one_hot_trit(b)])
            if op=='AND':
                y = min(a,b)
            elif op=='OR':
                y = max(a,b)
            elif op=='ADD':
                s = a+b
                s = max(-1, min(1, s))
                y = s
            else:
                raise ValueError(op)
            t = { -1:0, 0:1, 1:2 }[y]  # class index
            yvec = np.zeros(3); yvec[t]=1
            inputs.append(x); targets.append(yvec)
    return np.array(inputs), np.array(targets)

def init_mlp(input_dim=6, hidden=16, n_layers=2, out_dim=3):
    weights, biases = [], []
    prev = input_dim
    for i in range(n_layers):
        W = np.random.randn(out_dim if i==n_layers-1 else hidden, prev).astype(np.float32)*0.5
        b = np.zeros((out_dim if i==n_layers-1 else hidden,), dtype=np.float32)
        weights.append(W); biases.append(b)
        prev = out_dim if i==n_layers-1 else hidden
    return weights, biases

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def train(op='AND', epochs=500, lr=0.05, outdir='models'):
    X,Y = prepare_dataset(op)
    W1 = np.random.randn(16, X.shape[1]).astype(np.float32)*0.1
    b1 = np.zeros(16, dtype=np.float32)
    W2 = np.random.randn(3,16).astype(np.float32)*0.1
    b2 = np.zeros(3, dtype=np.float32)
    for e in range(epochs):
        # forward
        h = np.tanh(X.dot(W1.T) + b1)
        logits = np.tanh(h.dot(W2.T) + b2)  # final tanh, we'll map to classes via argmax
        probs = softmax(logits)
        loss = -np.sum(Y * np.log(probs + 1e-9)) / X.shape[0]
        # simple gradient descent on logits (not mathematically perfect but works for demo)
        grad_logits = (probs - Y) / X.shape[0]
        # backprop to W2,b2
        grad_W2 = grad_logits.T.dot(h)
        grad_b2 = grad_logits.sum(axis=0)
        # backprop to h
        grad_h = grad_logits.dot(W2) * (1 - h*h)
        grad_W1 = grad_h.T.dot(X)
        grad_b1 = grad_h.sum(axis=0)
        # update
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        if e % 100 == 0:
            acc = (np.argmax(probs, axis=1) == np.argmax(Y, axis=1)).mean()
            print(f"Epoch {e} loss={loss:.4f} acc={acc:.3f}")
    # save model
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    path = p / f"{op.lower()}_v1.npz"
    np.savez_compressed(path, n_layers=2, W0=W1, b0=b1, W1=W2, b1=b2)
    print("Saved", path)

if __name__ == '__main__':
    train('AND', epochs=1000, lr=0.02)
    train('OR', epochs=1000, lr=0.02)
    train('ADD', epochs=1200, lr=0.02)
