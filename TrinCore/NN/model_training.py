import numpy as np
from .nn import TernaryGateNN
from typing import Tuple, Dict, Any, List

def generate_training_data(operation: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for ternary operations"""
    X = []
    Y = []
    
    for a in range(3):
        for b in range(3):
            X.append([a, b])
            
            if operation == "ADD":
                result = (a + b) % 3
            elif operation == "SUB":
                result = (a - b) % 3
            elif operation == "AND":
                result = ternary_and(a, b)
            elif operation == "OR":
                result = ternary_or(a, b)
            elif operation == "XOR":
                result = ternary_xor(a, b)
            elif operation == "NAND":
                result = ternary_nand(a, b)
            elif operation == "NOR":
                result = ternary_nor(a, b)
            else:
                result = 0
            
            Y.append(result)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    Y_onehot = np.eye(3)[Y]
    
    return X, Y_onehot

def train_model_for_operation(operation: str, input_size: int = 2, 
                            output_size: int = 3, hidden_layers: List[int] = None,
                            learning_rate: float = 0.01, epochs: int = 1000) -> Tuple[TernaryGateNN, Dict[str, Any]]:
    """Train a neural network for a specific ternary operation"""
    if hidden_layers is None:
        hidden_layers = [16, 12, 8]
    
    # Generate training data
    X, Y = generate_training_data(operation)
    
    # Create and train model
    model = TernaryGateNN(
        input_neurons=input_size,
        hidden1=hidden_layers[0],
        hidden2=hidden_layers[1],
        hidden3=hidden_layers[2],
        output_neurons=output_size,
        lr=learning_rate
    )
    
    # Train model
    training_result = model.train(X, Y, epochs=epochs, patience=30, verbose=False)
    
    # Evaluate performance
    predictions = model.predict(X)
    accuracy = np.mean(predictions == np.argmax(Y, axis=1))
    
    return model, {
        'operation': operation,
        'accuracy': accuracy,
        'training_loss': training_result['final_total_loss'],
        'best_loss': training_result['best_loss'],
        'epochs': training_result['total_epochs']
    }
