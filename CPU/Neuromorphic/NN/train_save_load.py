import numpy as np
import pickle
import os
import json
from typing import Dict, Any, List
from TernaryNeuralCPU_nn import TernaryGateNN, GeneticNeuralModel

def save_model(model: TernaryGateNN, filepath: str):
    """Save a model's weights to a file"""
    weights_data = model.get_weights()
    
    # Flatten the nested structure for numpy save
    save_data = {}
    for key, value in weights_data['weights'].items():
        save_data[key] = value
    for key, value in weights_data['biases'].items():
        save_data[key] = value
    
    np.savez(filepath, **save_data)
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
    print(f"Model loaded from {filepath}")
    return model

def save_training_history(history: Dict[str, Any], filepath: str):
    """Save training history to a JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    json_compatible_history = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            json_compatible_history[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_compatible_history[key] = value.item()
        else:
            json_compatible_history[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_compatible_history, f, indent=2)
    print(f"Training history saved to {filepath}")

def load_training_history(filepath: str) -> Dict[str, Any]:
    """Load training history from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_genetic_model(model: GeneticNeuralModel, filepath: str):
    """Save a genetic model to a file"""
    data = {
        'architecture': model.architecture,
        'weights': {k: v.copy() for k, v in model.weights.items()},
        'biases': {k: v.copy() for k, v in model.biases.items()},
        'mutation_rate': model.mutation_rate,
        'generation': model.generation,
        'fitness': model.fitness,
        'training_history': model.training_history
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Genetic model saved to {filepath}")

def load_genetic_model(filepath: str) -> GeneticNeuralModel:
    """Load a genetic model from a file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = GeneticNeuralModel(data['architecture'], data['mutation_rate'])
    model.weights = data['weights']
    model.biases = data['biases']
    model.generation = data['generation']
    model.fitness = data['fitness']
    model.training_history = data['training_history']
    
    print(f"Genetic model loaded from {filepath}")
    return model

def create_model_directory(base_path: str = "saved_models"):
    """Create directory structure for saved models"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "standard"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "genetic"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "history"), exist_ok=True)
    return base_path

def get_model_filepath(operation: str, model_type: str = "standard", base_path: str = "saved_models") -> str:
    """Generate standardized filepath for models"""
    create_model_directory(base_path)
    if model_type == "genetic":
        return os.path.join(base_path, "genetic", f"{operation}_genetic_model.pkl")
    else:
        return os.path.join(base_path, "standard", f"{operation}_model.npz")

def save_best_model(model, operation: str, performance_metrics: Dict[str, Any], 
                   model_type: str = "standard", base_path: str = "saved_models"):
    """Save model only if it's better than existing one"""
    filepath = get_model_filepath(operation, model_type, base_path)
    history_path = os.path.join(base_path, "history", f"{operation}_{model_type}_best.json")
    
    # Check if we should save this model
    should_save = True
    if os.path.exists(history_path):
        try:
            existing_metrics = load_training_history(history_path)
            # Compare accuracy (higher is better)
            if existing_metrics.get('accuracy', 0) >= performance_metrics.get('accuracy', 0):
                should_save = False
                print(f"Existing model for {operation} is better. Not saving.")
        except:
            should_save = True
    
    if should_save:
        if model_type == "genetic":
            save_genetic_model(model, filepath)
        else:
            save_model(model, filepath)
        save_training_history(performance_metrics, history_path)
        print(f"New best model saved for {operation}!")
    
    return should_save
