#!/usr/bin/env python3
"""
Fixed and Enhanced Model Training System
Corrects all bugs and implements proper neural network training
"""

import numpy as np
import time
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

# Fixed ternary operations with proper validation
def ternary_and(a: int, b: int) -> int:
    """Ternary AND operation with validation"""
    a, b = int(a), int(b)
    if not (0 <= a <= 2 and 0 <= b <= 2):
        raise ValueError(f"Invalid ternary values: {a}, {b}")
    
    if a == 0 or b == 0:
        return 0
    elif a == 2 and b == 2:
        return 2
    else:
        return 1

def ternary_or(a: int, b: int) -> int:
    """Ternary OR operation"""
    a, b = int(a), int(b)
    if not (0 <= a <= 2 and 0 <= b <= 2):
        raise ValueError(f"Invalid ternary values: {a}, {b}")
    return max(a, b)

def ternary_xor(a: int, b: int) -> int:
    """Ternary XOR operation"""
    a, b = int(a), int(b)
    if not (0 <= a <= 2 and 0 <= b <= 2):
        raise ValueError(f"Invalid ternary values: {a}, {b}")
    return (a + b) % 3 if a != b else 0

def ternary_nand(a: int, b: int) -> int:
    """Ternary NAND operation"""
    return 2 - ternary_and(a, b)

def ternary_nor(a: int, b: int) -> int:
    """Ternary NOR operation"""
    return 2 - ternary_or(a, b)

def ternary_add(a: int, b: int) -> int:
    """Ternary ADD operation"""
    a, b = int(a), int(b)
    if not (0 <= a <= 2 and 0 <= b <= 2):
        raise ValueError(f"Invalid ternary values: {a}, {b}")
    return (a + b) % 3

def ternary_sub(a: int, b: int) -> int:
    """Ternary SUB operation"""
    a, b = int(a), int(b)
    if not (0 <= a <= 2 and 0 <= b <= 2):
        raise ValueError(f"Invalid ternary values: {a}, {b}")
    return (a - b) % 3

# Operation mapping
OPERATIONS = {
    'AND': ternary_and,
    'OR': ternary_or,
    'XOR': ternary_xor,
    'NAND': ternary_nand,
    'NOR': ternary_nor,
    'ADD': ternary_add,
    'SUB': ternary_sub
}

class FixedTernaryGateNN:
    """Fixed ternary neural network with proper training"""
    
    def __init__(self, input_neurons: int = 2, hidden1: int = 32, hidden2: int = 24, 
                 hidden3: int = 16, output_neurons: int = 3, lr: float = 0.01):
        self.input_neurons = input_neurons
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_neurons = output_neurons
        self.lr = lr
        
        # Proper Xavier/He initialization
        self.weights = {
            'w1': np.random.randn(self.input_neurons, self.hidden1) * np.sqrt(2.0 / self.input_neurons),
            'w2': np.random.randn(self.hidden1, self.hidden2) * np.sqrt(2.0 / self.hidden1),
            'w3': np.random.randn(self.hidden2, self.hidden3) * np.sqrt(2.0 / self.hidden2),
            'w4': np.random.randn(self.hidden3, self.output_neurons) * np.sqrt(2.0 / self.hidden3)
        }
        
        # Initialize biases to small positive values
        self.biases = {
            'b1': np.random.normal(0, 0.01, (1, self.hidden1)),
            'b2': np.random.normal(0, 0.01, (1, self.hidden2)),
            'b3': np.random.normal(0, 0.01, (1, self.hidden3)),
            'b4': np.zeros((1, self.output_neurons))  # Output bias stays zero
        }
        
        # Training history
        self.training_history = []
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(np.float32)
    
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation"""
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1.0, alpha)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation"""
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(np.clip(x - x_max, -500, 500))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)
    
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass with proper caching"""
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(x, self.weights['w1']) + self.biases['b1']
        a1 = self.leaky_relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['w2']) + self.biases['b2']
        a2 = self.leaky_relu(z2)
        
        # Layer 3
        z3 = np.dot(a2, self.weights['w3']) + self.biases['b3']
        a3 = self.leaky_relu(z3)
        
        # Output layer
        z4 = np.dot(a3, self.weights['w4']) + self.biases['b4']
        a4 = self.softmax(z4)
        
        return {
            'x': x,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3,
            'z4': z4, 'a4': a4
        }
    
    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass with proper gradient computation"""
        m = cache['x'].shape[0]
        
        # Output layer gradients
        dz4 = cache['a4'] - y_true
        dw4 = np.dot(cache['a3'].T, dz4) / m
        db4 = np.sum(dz4, axis=0, keepdims=True) / m
        
        # Layer 3 gradients
        da3 = np.dot(dz4, self.weights['w4'].T)
        dz3 = da3 * self.leaky_relu_derivative(cache['z3'])
        dw3 = np.dot(cache['a2'].T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Layer 2 gradients
        da2 = np.dot(dz3, self.weights['w3'].T)
        dz2 = da2 * self.leaky_relu_derivative(cache['z2'])
        dw2 = np.dot(cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Layer 1 gradients
        da1 = np.dot(dz2, self.weights['w2'].T)
        dz1 = da1 * self.leaky_relu_derivative(cache['z1'])
        dw1 = np.dot(cache['x'].T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {
            'dw1': dw1, 'db1': db1,
            'dw2': dw2, 'db2': db2,
            'dw3': dw3, 'db3': db3,
            'dw4': dw4, 'db4': db4
        }
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """Update weights using gradients with gradient clipping"""
        # Gradient clipping
        max_grad_norm = 5.0
        
        for key in gradients:
            grad_norm = np.linalg.norm(gradients[key])
            if grad_norm > max_grad_norm:
                gradients[key] = gradients[key] * (max_grad_norm / grad_norm)
        
        # Update weights
        self.weights['w1'] -= learning_rate * gradients['dw1']
        self.biases['b1'] -= learning_rate * gradients['db1']
        self.weights['w2'] -= learning_rate * gradients['dw2']
        self.biases['b2'] -= learning_rate * gradients['db2']
        self.weights['w3'] -= learning_rate * gradients['dw3']
        self.biases['b3'] -= learning_rate * gradients['db3']
        self.weights['w4'] -= learning_rate * gradients['dw4']
        self.biases['b4'] -= learning_rate * gradients['db4']
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get predictions"""
        cache = self.forward(x)
        return np.argmax(cache['a4'], axis=1)
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 2000, 
              batch_size: int = None, validation_split: float = 0.0,
              early_stopping: bool = True, patience: int = 100) -> Dict[str, Any]:
        """Train the neural network with proper training loop"""
        
        # Convert Y to one-hot if needed
        if Y.ndim == 1:
            Y_onehot = np.eye(3)[Y]
        else:
            Y_onehot = Y
            Y = np.argmax(Y, axis=1)
        
        # Split data if validation is requested
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            Y_train, Y_val = Y_onehot[:split_idx], Y_onehot[split_idx:]
            Y_train_labels, Y_val_labels = Y[:split_idx], Y[split_idx:]
        else:
            X_train, Y_train, Y_train_labels = X, Y_onehot, Y
            X_val = Y_val = Y_val_labels = None
        
        # Training parameters
        if batch_size is None:
            batch_size = len(X_train)
        
        best_loss = float('inf')
        best_accuracy = 0.0
        patience_counter = 0
        
        # Adaptive learning rate
        current_lr = self.lr
        lr_decay = 0.95
        lr_patience = 20
        lr_counter = 0
        
        print(f"Training with {len(X_train)} samples, {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            # Mini-batch training
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward pass
                cache = self.forward(X_batch)
                
                # Compute loss
                epsilon = 1e-8
                loss = -np.mean(np.sum(Y_batch * np.log(cache['a4'] + epsilon), axis=1))
                total_loss += loss
                num_batches += 1
                
                # Backward pass
                gradients = self.backward(cache, Y_batch)
                
                # Update weights
                self.update_weights(gradients, current_lr)
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Evaluate on training set
            train_pred = self.predict(X_train)
            train_accuracy = np.mean(train_pred == Y_train_labels)
            
            # Evaluate on validation set if available
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_accuracy = np.mean(val_pred == Y_val_labels)
                val_cache = self.forward(X_val)
                val_loss = -np.mean(np.sum(Y_val * np.log(val_cache['a4'] + 1e-8), axis=1))
            else:
                val_accuracy = train_accuracy
                val_loss = avg_loss
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': current_lr
            })
            
            # Early stopping and learning rate adjustment
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                lr_counter = 0
            else:
                patience_counter += 1
                lr_counter += 1
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            
            # Reduce learning rate on plateau
            if lr_counter >= lr_patience:
                current_lr *= lr_decay
                lr_counter = 0
                if current_lr < 1e-6:
                    current_lr = 1e-6
                print(f"    Learning rate reduced to {current_lr:.6f}")
            
            # Progress reporting
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}: Loss={avg_loss:.6f}, "
                      f"Train Acc={train_accuracy:.6f}, Val Acc={val_accuracy:.6f}, "
                      f"LR={current_lr:.6f}")
            
            # Early stopping
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break
            
            # Perfect accuracy stopping
            if val_accuracy >= 0.9999:
                print(f"Perfect accuracy achieved at epoch {epoch}")
                break
        
        # Final evaluation
        final_pred = self.predict(X)
        final_accuracy = np.mean(final_pred == Y)
        
        training_results = {
            'final_accuracy': final_accuracy,
            'best_accuracy': best_accuracy,
            'final_loss': avg_loss,
            'best_loss': best_loss,
            'epochs_trained': epoch + 1,
            'training_history': self.training_history,
            'perfect_accuracy': final_accuracy >= 0.9999
        }
        
        return training_results
    
    def get_weights(self) -> Dict[str, Any]:
        """Get model weights"""
        return {
            'weights': self.weights.copy(),
            'biases': self.biases.copy(),
            'architecture': {
                'input_neurons': self.input_neurons,
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'hidden3': self.hidden3,
                'output_neurons': self.output_neurons
            }
        }
    
    def set_weights(self, weights_data: Dict[str, Any]):
        """Set model weights"""
        self.weights = weights_data['weights'].copy()
        self.biases = weights_data['biases'].copy()

class EnhancedGeneticNeuralModel:
    """Enhanced genetic model with proper implementation"""
    
    def __init__(self, architecture: List[int], mutation_rate: float = 0.3):
        self.architecture = architecture
        self.mutation_rate = mutation_rate
        self.fitness = 0.0
        self.accuracy = 0.0
        self.generation = 0
        self.training_history = []
        
        # Initialize weights and biases properly
        self.weights = {}
        self.biases = {}
        
        for i in range(len(architecture) - 1):
            fan_in = architecture[i]
            fan_out = architecture[i + 1]
            # He initialization for ReLU-like activations
            self.weights[f'w{i}'] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.biases[f'b{i}'] = np.random.normal(0, 0.01, (1, fan_out))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with improved activation"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        a = X.astype(np.float32)
        
        for i in range(len(self.architecture) - 1):
            z = np.dot(a, self.weights[f'w{i}']) + self.biases[f'b{i}']
            
            if i == len(self.architecture) - 2:  # Output layer
                # Stable softmax
                z_max = np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(np.clip(z - z_max, -500, 500))
                a = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
            else:
                # Leaky ReLU for hidden layers
                a = np.where(z > 0, z, 0.01 * z)
        
        return a
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def mutate(self, mutation_rate: Optional[float] = None) -> 'EnhancedGeneticNeuralModel':
        """Create mutated copy with improved mutation"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        child = EnhancedGeneticNeuralModel(self.architecture.copy(), mutation_rate)
        child.generation = self.generation + 1
        
        # Copy and mutate weights
        for key in self.weights:
            child.weights[key] = self.weights[key].copy()
            if np.random.random() < mutation_rate:
                mutation_strength = 0.1 * (1.0 - self.accuracy)  # Adaptive strength
                noise = np.random.normal(0, mutation_strength, self.weights[key].shape)
                child.weights[key] += noise.astype(np.float32)
        
        # Copy and mutate biases
        for key in self.biases:
            child.biases[key] = self.biases[key].copy()
            if np.random.random() < mutation_rate:
                mutation_strength = 0.05 * (1.0 - self.accuracy)  # Adaptive strength
                noise = np.random.normal(0, mutation_strength, self.biases[key].shape)
                child.biases[key] += noise.astype(np.float32)
        
        return child
    
    def crossover(self, partner: 'EnhancedGeneticNeuralModel') -> 'EnhancedGeneticNeuralModel':
        """Create offspring through crossover"""
        child = EnhancedGeneticNeuralModel(self.architecture.copy(), self.mutation_rate)
        child.generation = max(self.generation, partner.generation) + 1
        
        # Crossover weights
        for key in self.weights:
            mask = np.random.random(self.weights[key].shape) < 0.5
            child.weights[key] = np.where(mask, self.weights[key], partner.weights[key])
        
        # Crossover biases
        for key in self.biases:
            mask = np.random.random(self.biases[key].shape) < 0.5
            child.biases[key] = np.where(mask, self.biases[key], partner.biases[key])
        
        return child

def generate_training_data(operation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete training data with validation"""
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")
    
    op_func = OPERATIONS[operation]
    X, Y, Y_onehot = [], [], []
    
    for a in range(3):
        for b in range(3):
            X.append([a, b])
            result = op_func(a, b)
            Y.append(result)
            
            # One-hot encoding
            y_vec = np.zeros(3)
            y_vec[result] = 1
            Y_onehot.append(y_vec)
    
    return (np.array(X, dtype=np.float32), 
            np.array(Y, dtype=np.int32), 
            np.array(Y_onehot, dtype=np.float32))

def train_standard_model_enhanced(operation: str, epochs: int = 3000, 
                                lr: float = 0.01, hidden_layers: List[int] = None) -> Tuple[FixedTernaryGateNN, Dict[str, Any]]:
    """Train enhanced standard neural network"""
    if hidden_layers is None:
        hidden_layers = [32, 24, 16]
    
    print(f"\n🧠 Training Enhanced Standard NN for {operation}")
    print("-" * 50)
    
    # Generate data
    X, Y, Y_onehot = generate_training_data(operation)
    
    # Create model
    model = FixedTernaryGateNN(
        input_neurons=2,
        hidden1=hidden_layers[0],
        hidden2=hidden_layers[1],
        hidden3=hidden_layers[2],
        output_neurons=3,
        lr=lr
    )
    
    # Train model
    start_time = time.time()
    results = model.train(
        X, Y_onehot, 
        epochs=epochs,
        batch_size=9,  # Use full dataset as batch
        validation_split=0.0,  # No validation split for small dataset
        early_stopping=True,
        patience=200
    )
    training_time = time.time() - start_time
    
    # Test final accuracy
    predictions = model.predict(X)
    final_accuracy = np.mean(predictions == Y)
    
    # Update results
    results.update({
        'operation': operation,
        'training_time': training_time,
        'model_type': 'standard_enhanced',
        'final_test_accuracy': final_accuracy
    })
    
    print(f"✅ Training complete: {final_accuracy:.6f} accuracy in {training_time:.2f}s")
    
    return model, results

def save_model_enhanced(model, operation: str, performance_metrics: Dict[str, Any], 
                       model_type: str = "standard"):
    """Save model with enhanced metadata"""
    models_dir = Path("models_enhanced")
    models_dir.mkdir(exist_ok=True)
    
    if model_type == "standard":
        # Save standard model
        weights_data = model.get_weights()
        filename = models_dir / f"{operation.lower()}_standard_enhanced.npz"
        
        save_data = {
            **weights_data['weights'],
            **weights_data['biases'],
            'accuracy': performance_metrics['final_test_accuracy'],
            'architecture': json.dumps(weights_data['architecture']),
            'training_time': performance_metrics['training_time'],
            'epochs': performance_metrics['epochs_trained']
        }
        
        np.savez_compressed(filename, **save_data)
        
    elif model_type == "genetic":
        # Save genetic model
        filename = models_dir / f"{operation.lower()}_genetic_enhanced.npz"
        
        save_data = {
            'architecture': json.dumps(model.architecture),
            'accuracy': model.accuracy,
            'fitness': model.fitness,
            'generation': model.generation
        }
        
        # Add weights and biases
        for key, value in model.weights.items():
            save_data[f'weights_{key}'] = value
        for key, value in model.biases.items():
            save_data[f'biases_{key}'] = value
        
        np.savez_compressed(filename, **save_data)
    
    # Save training history
    history_file = models_dir / f"{operation.lower()}_{model_type}_history.json"
    with open(history_file, 'w') as f:
        json.dump(performance_metrics, f, indent=2, default=str)
    
    print(f"💾 Saved {operation} {model_type} model to {filename}")

def test_model_thoroughly(model, operation: str, model_type: str = "standard") -> bool:
    """Thoroughly test a trained model"""
    X, Y, Y_onehot = generate_training_data(operation)
    
    if model_type == "standard":
        predictions = model.predict(X)
    else:  # genetic
        predictions = model.predict(X)
    
    accuracy = np.mean(predictions == Y)
    
    print(f"\n🧪 THOROUGH TEST for {operation} ({model_type}):")
    print("Input | Expected | Predicted | Correct | Confidence")
    print("-" * 55)
    
    all_correct = True
    confidences = []
    
    if model_type == "standard":
        probs = model.forward(X)['a4']
    else:
        probs = model.forward(X)
    
    for i, ((a, b), expected, predicted) in enumerate(zip(X, Y, predictions)):
        correct = "✓" if expected == predicted else "✗"
        confidence = probs[i][predicted]
        confidences.append(confidence)
        
        if expected != predicted:
            all_correct = False
        
        print(f"({int(a)},{int(b)}) |    {expected}     |     {predicted}     |   {correct}   |   {confidence:.4f}")
    
    avg_confidence = np.mean(confidences)
    print(f"\nAccuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Perfect: {'YES' if all_correct else 'NO'}")
    
    return all_correct

def train_all_operations_enhanced():
    """Train enhanced models for all operations"""
    print("🚀 ENHANCED MODEL TRAINING - TARGETING 100% ACCURACY")
    print("=" * 70)
    
    operations = ['AND', 'OR', 'XOR', 'ADD', 'SUB', 'NAND', 'NOR']
    results = {}
    perfect_count = 0
    
    for operation in operations:
        print(f"\n{'='*25} {operation} {'='*25}")
        
        try:
            # Train standard model
            model, metrics = train_standard_model_enhanced(
                operation, 
                epochs=4000, 
                lr=0.02,
                hidden_layers=[40, 32, 24]
            )
            
            # Test thoroughly
            is_perfect = test_model_thoroughly(model, operation, "standard")
            
            if is_perfect:
                perfect_count += 1
                save_model_enhanced(model, operation, metrics, "standard")
            
            results[operation] = {
                'accuracy': metrics['final_test_accuracy'],
                'perfect': is_perfect,
                'epochs': metrics['epochs_trained'],
                'time': metrics['training_time']
            }
            
            status = "🎯 PERFECT" if is_perfect else f"📊 {metrics['final_test_accuracy']:.4f}"
            print(f"✅ {operation}: {status}")
            
        except Exception as e:
            print(f"❌ {operation}: ERROR - {e}")
            results[operation] = {'accuracy': 0.0, 'perfect': False}
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 ENHANCED TRAINING RESULTS")
    print("=" * 70)
    
    for operation, result in results.items():
        if 'accuracy' in result:
            status_icon = "🎯" if result['perfect'] else "📊"
            acc_str = "100.0000%" if result['perfect'] else f"{result['accuracy']*100:.4f}%"
            time_str = f"{result.get('time', 0):.1f}s" if 'time' in result else "N/A"
            epoch_str = f"E{result.get('epochs', 0)}" if 'epochs' in result else "N/A"
            print(f"{status_icon} {operation:4s}: {acc_str:>10s} | {time_str:>6s} | {epoch_str:>5s}")
        else:
            print(f"❌ {operation:4s}: ERROR")
    
    print(f"\nPERFECT MODELS: {perfect_count}/{len(operations)}")
    
    if perfect_count == len(operations):
        print("\n🎉 ALL OPERATIONS ACHIEVED 100% ACCURACY! 🎉")
    else:
        print(f"\n⚡ {perfect_count} operations achieved 100% accuracy")

if __name__ == '__main__':
    train_all_operations_enhanced()
