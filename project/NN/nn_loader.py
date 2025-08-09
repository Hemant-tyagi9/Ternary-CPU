import numpy as np
from pathlib import Path

class NeuroGate:
    def __init__(self, weights, biases):
        # weights: list of 2D arrays, biases: list of 1D arrays
        self.weights = weights
        self.biases = biases
    def infer(self, batch_x):
        # batch_x: list or array of shape (N, input_dim)
        x = np.array(batch_x, dtype=np.float32)
        for W,b in zip(self.weights, self.biases):
            x = x.dot(W.T) + b
            # hidden activations: tanh, last layer linear (we will apply argmax outside)
            x = np.tanh(x)
        # output layer: produce 3 logits per sample (we used final tanh as well)
        return x

def load_npz_model(path):
    data = np.load(path)
    weights = [data[f"W{i}"] for i in range(data["n_layers"])]
    biases = [data[f"b{i}"] for i in range(data["n_layers"])]
    return NeuroGate(weights, biases)

def load_models_from_dir(dirpath="models"):
    p = Path(dirpath)
    mapping = {}
    if not p.exists():
        return mapping
    for f in p.glob("*.npz"):
        name = f.stem.split('_')[0].upper()
        mapping[name] = load_npz_model(f)
    return mapping
