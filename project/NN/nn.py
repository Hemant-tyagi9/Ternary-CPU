import numpy as np
from typing import List, Dict, Any


class TernaryGateNN:
    """Ternary neural network for gate operations"""
    
    def __init__(self, input_neurons=2, hidden1=16, hidden2=12, hidden3=8, 
                 output_neurons=3, lr=0.01, dropout_rate=0.1, l2_reg=0.001):
        self.input_neurons = input_neurons
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_neurons = output_neurons
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.training_history = []

        self.weights = {
            'w1': np.random.randn(self.input_neurons, self.hidden1) * np.sqrt(2./self.input_neurons),
            'w2': np.random.randn(self.hidden1, self.hidden2) * np.sqrt(2./self.hidden1),
            'w3': np.random.randn(self.hidden2, self.hidden3) * np.sqrt(2./self.hidden2),
            'w4': np.random.randn(self.hidden3, self.output_neurons) * np.sqrt(2./self.hidden3)
        }
        
        # Initialize biases
        self.biases = {
            'b1': np.zeros((1, self.hidden1)),
            'b2': np.zeros((1, self.hidden2)),
            'b3': np.zeros((1, self.hidden3)),
            'b4': np.zeros((1, self.output_neurons))
        }

    def batch_norm(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
    
    # Keep all other methods the same...
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        # Forward pass
        z1 = np.dot(x, self.weights['w1']) + self.biases['b1']
        a1 = self.sigmoid(z1)
        
        z2 = np.dot(a1, self.weights['w2']) + self.biases['b2']
        a2 = self.sigmoid(z2)
        
        z3 = np.dot(a2, self.weights['w3']) + self.biases['b3']
        a3 = self.sigmoid(z3)
        
        z4 = np.dot(a3, self.weights['w4']) + self.biases['b4']
        a4 = self.softmax(z4)
        
        return {
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3,
            'z4': z4, 'a4': a4
        }
    
    def backward(self, x: np.ndarray, y: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        m = x.shape[0]
        
        # Calculate gradients
        dz4 = cache['a4'] - y
        dw4 = np.dot(cache['a3'].T, dz4) / m
        db4 = np.sum(dz4, axis=0, keepdims=True) / m
        
        da3 = np.dot(dz4, self.weights['w4'].T)
        dz3 = da3 * cache['a3'] * (1 - cache['a3'])
        dw3 = np.dot(cache['a2'].T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        da2 = np.dot(dz3, self.weights['w3'].T)
        dz2 = da2 * cache['a2'] * (1 - cache['a2'])
        dw2 = np.dot(cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.weights['w2'].T)
        dz1 = da1 * cache['a1'] * (1 - cache['a1'])
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {
            'dw1': dw1, 'db1': db1,
            'dw2': dw2, 'db2': db2,
            'dw3': dw3, 'db3': db3,
            'dw4': dw4, 'db4': db4
        }
    
    def update_parameters(self, grads: Dict[str, np.ndarray]):
        # Update weights and biases
        for key in self.weights:
            self.weights[key] -= self.lr * grads[f'd{key}']
        for key in self.biases:
            self.biases[key] -= self.lr * grads[f'd{key}']
    
    def compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        # Cross-entropy loss
        m = y.shape[0]
        log_likelihood = -np.log(y_hat[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              patience: int = 30, verbose: bool = True) -> Dict[str, Any]:
        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        
        for epoch in range(epochs):
            # Forward and backward pass
            cache = self.forward(x)
            grads = self.backward(x, y, cache)
            self.update_parameters(grads)
            
            # Calculate loss
            loss = self.compute_loss(y, cache['a4'])
            loss_history.append(loss)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.training_history.extend(loss_history)
        return {
            'final_total_loss': loss,
            'best_loss': best_loss,
            'total_epochs': epoch + 1
        }
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        cache = self.forward(x)
        return np.argmax(cache['a4'], axis=1)
    
    def get_loss_summary(self) -> Dict[str, Any]:
        if not self.training_history:
            return {}
        return {
            'final_total_loss': self.training_history[-1],
            'best_loss': min(self.training_history),
            'total_epochs': len(self.training_history)
        }
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            'weights': {k: v.copy() for k, v in self.weights.items()},
            'biases': {k: v.copy() for k, v in self.biases.items()}
        }
    
    def set_weights(self, weights_data: Dict[str, Any]):
        if 'weights' in weights_data:
            self.weights = {k: v.copy() for k, v in weights_data['weights'].items()}
        if 'biases' in weights_data:
            self.biases = {k: v.copy() for k, v in weights_data['biases'].items()}
            # Add regularization
    def batch_norm(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)



class ModelMetrics:
    accuracy: float
    loss: float
    generation: int
    fitness_score: float
    training_time: float
    mutation_rate: float
    architecture: List[int]
    timestamp: str

class GeneticNeuralModel:
    """Genetically evolved neural network for ternary operations"""
    
    def __init__(self, architecture: List[int], mutation_rate: float = 0.1):
        self.architecture = architecture
        self.mutation_rate = mutation_rate
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        self.fitness = 0.0
        self.generation = 0
        self.training_history = []
        
    def _initialize_weights(self):
        weights = {}
        for i in range(len(self.architecture) - 1):
            key = f'w{i+1}'
            weights[key] = np.random.randn(
                self.architecture[i], 
                self.architecture[i+1]
            ) * 0.5
        return weights
    
    def _initialize_biases(self):
        biases = {}
        for i in range(1, len(self.architecture)):
            key = f'b{i}'
            biases[key] = np.zeros((1, self.architecture[i]))
        return biases
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.architecture) - 1):
            z = np.dot(activations[-1], self.weights[f'w{i+1}']) + self.biases[f'b{i+1}']
            if i == len(self.architecture) - 2:  # Output layer
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations[-1]
    
    def mutate(self):
        """Apply genetic mutations to the model"""
        new_model = GeneticNeuralModel(self.architecture.copy(), self.mutation_rate)
        
        # Mutate weights
        for key in self.weights:
            new_model.weights[key] = self.weights[key].copy()
            if np.random.random() < self.mutation_rate:
                mutation_mask = np.random.random(self.weights[key].shape) < 0.3
                new_model.weights[key][mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        
        # Mutate biases
        for key in self.biases:
            new_model.biases[key] = self.biases[key].copy()
            if np.random.random() < self.mutation_rate:
                new_model.biases[key] += np.random.normal(0, 0.05, self.biases[key].shape)
        
        new_model.generation = self.generation + 1
        return new_model
    
    def crossover(self, partner):
        """Create offspring through genetic crossover"""
        child = GeneticNeuralModel(self.architecture.copy(), self.mutation_rate)
        
        # Weight crossover
        for key in self.weights:
            mask = np.random.random(self.weights[key].shape) < 0.5
            child.weights[key] = np.where(mask, self.weights[key], partner.weights[key])
        
        # Bias crossover
        for key in self.biases:
            mask = np.random.random(self.biases[key].shape) < 0.5
            child.biases[key] = np.where(mask, self.biases[key], partner.biases[key])
        
        child.generation = max(self.generation, partner.generation) + 1
        return child
