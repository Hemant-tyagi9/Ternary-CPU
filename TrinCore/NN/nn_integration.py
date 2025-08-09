import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from trincore_applications.config import TrinCoreConfig
from trincore_applications.logging import logger


class NeuralIntegration:
    """Enhanced neural network integration with vectorized operations"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.config = TrinCoreConfig()
        self._setup_gate_operations()
    
    def _setup_gate_operations(self):
        """Initialize gate operation implementations"""
        self.gate_operations = {
            "AND": self._ternary_and,
            "OR": self._ternary_or,
            "XOR": self._ternary_xor,
            "ADD": self._ternary_add,
            "SUB": self._ternary_sub
        }
    
    def train_models(self, operations: List[str]):
        """Train models for specified operations"""
        for op in operations:
            if op not in self.models:
                # In a real implementation, this would train or load the model
                self.models[op] = self._create_dummy_model(op)
                logger.info(f"Initialized model for {op} operation")
    
    def _create_dummy_model(self, operation: str):
        """Create a dummy model that uses the gate logic"""
        class DummyModel:
            def __init__(self, op):
                self.op = op
            
            def predict(self, X):
                if self.op == "AND":
                    return np.vectorize(ternary_and_balanced)(X[:,0], X[:,1])
                elif self.op == "OR":
                    return np.vectorize(lambda a,b: max(a,b))(X[:,0], X[:,1])
                # Add other operations as needed
                return np.zeros(X.shape[0])
        
        return DummyModel(operation)
    
    def execute_operation(self, operation: str, a: int, b: int) -> int:
        """Execute a single operation"""
        if operation in self.models and self.config.get('nn.vectorize_ops', True):
            # Use vectorized version even for single operations
            result = self.execute_batch_operations(operation, [(a, b)])[0]
        elif operation in self.gate_operations:
            result = self.gate_operations[operation](a, b)
        else:
            result = 0
        
        logger.log_operation(operation, (a, b), result)
        return result
    
    def execute_batch_operations(self, operation: str, operands: List[Tuple[int, int]]) -> np.ndarray:
        """Execute multiple operations in batch for efficiency"""
        if operation not in self.models:
            if operation in self.gate_operations:
                # Fallback to vectorized gate operations
                a_vals = np.array([op[0] for op in operands])
                b_vals = np.array([op[1] for op in operands])
                return np.vectorize(self.gate_operations[operation])(a_vals, b_vals)
            return np.zeros(len(operands))
        
        # Prepare input data
        input_data = np.array(operands)
        
        # Predict using the model
        predictions = self.models[operation].predict(input_data)
        
        # Convert to proper ternary values
        return np.clip(np.round(predictions), 0, 2).astype(int)
    
    # Gate operation implementations
    def _ternary_and(self, a: int, b: int) -> int:
        return ternary_and_balanced(a, b)
    
    def _ternary_or(self, a: int, b: int) -> int:
        return max(a, b)
    
    def _ternary_xor(self, a: int, b: int) -> int:
        return (a - b) % 3 if a != b else 0
    
    def _ternary_add(self, a: int, b: int) -> int:
        return (a + b) % 3
    
    def _ternary_sub(self, a: int, b: int) -> int:
        return (a - b) % 3

def ternary_and_balanced(a: int, b: int) -> int:
    """Balanced ternary AND operation"""
    lookup = {
        (0,0):0, (0,1):0, (0,2):0,
        (1,0):0, (1,1):1, (1,2):1,
        (2,0):0, (2,1):1, (2,2):2
    }
    return lookup.get((a,b), 0)
    
    
def visualize_genetic_training(history: List[Dict[str, Any]], operation: str):
    """Visualize genetic training progress"""
    plt.figure(figsize=(10, 6))
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    accuracy = [h['best_accuracy'] for h in history]
    
    plt.plot(generations, best_fitness, label='Best Fitness')
    plt.plot(generations, avg_fitness, label='Average Fitness')
    plt.plot(generations, accuracy, label='Accuracy', linestyle='--')
    
    plt.title(f'Genetic Training Progress - {operation}')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
