import numpy as np
import os
import json
from typing import Dict, Any, List
from .nn import TernaryGateNN

def save_model(model: TernaryGateNN, filepath: str):
    """Save a model's weights to a file"""
    weights = model.get_weights()
    np.savez(filepath, **weights)
    print(f"Model saved to {filepath}")

def load_model(filepath: str, input_size: int = 2, hidden_layers: List[int] = None,
              output_size: int = 3, learning_rate: float = 0.01) -> TernaryGateNN:
    """Load a model from a weights file"""
    if hidden_layers is None:
        hidden_layers = [16, 12, 8]
    
    # Create new model with same architecture
    model = TernaryGateNN(
        input_neurons=input_size,
        hidden1=hidden_layers[0],
        hidden2=hidden_layers[1],
        hidden3=hidden_layers[2],
        output_neurons=output_size,
        lr=learning_rate
    )
    
    # Load weights
    weights_data = np.load(filepath)
    weights = {
        'weights': {
            'w1': weights_data['w1'],
            'w2': weights_data['w2'],
            'w3': weights_data['w3'],
            'w4': weights_data['w4']
        },
        'biases': {
            'b1': weights_data['b1'],
            'b2': weights_data['b2'],
            'b3': weights_data['b3'],
            'b4': weights_data['b4']
        }
    }
    
    model.set_weights(weights)
    return model

def save_training_history(history: Dict[str, Any], filepath: str):
    """Save training history to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {filepath}")

def load_training_history(filepath: str) -> Dict[str, Any]:
    """Load training history from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
