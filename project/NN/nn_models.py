import os
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .nn import TernaryGateNN, GeneticNeuralModel
from .model_training import train_or_load_model, train_genetic_model_advanced, generate_training_data
from .train_save_load import save_best_model, load_model, load_genetic_model, get_model_filepath
from .nn_benchmark import benchmark_neural_operation, benchmark_genetic_model


@dataclass
class ModelPerformance:
    """Data class for storing model performance metrics"""
    accuracy: float
    avg_inference_time: float
    training_time: float
    model_size: int
    generation: int = 0
    fitness_score: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TernaryNeuralCPU:
    """
    Main class representing the neural processing unit of the ternary CPU
    Manages all neural models and provides intelligent operation execution
    """
    
    def __init__(self, auto_optimize: bool = True, cache_models: bool = True):
        self.auto_optimize = auto_optimize
        self.cache_models = cache_models
        self.models: Dict[str, Dict[str, Any]] = {}  # operation -> {standard: model, genetic: model}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.supported_operations = ["AND", "OR", "XOR", "NAND", "NOR", "ADD", "SUB"]
        
        # Performance thresholds
        self.accuracy_threshold = 0.95
        self.inference_time_threshold = 0.001  # 1ms
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the neural CPU system"""
        print("Initializing TernaryNeuralCPU...")
        
        # Create model directories
        from train_save_load import create_model_directory
        create_model_directory()
        
        # Load performance history
        self._load_performance_history()
        
        # Initialize operation counters
        for op in self.supported_operations:
            self.operation_counts[op] = 0
            if op not in self.performance_history:
                self.performance_history[op] = []
        
        print(f"Neural CPU initialized with support for: {', '.join(self.supported_operations)}")
    
    def _load_performance_history(self):
        """Load historical performance data"""
        history_file = "saved_models/performance_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for op, history in data.items():
                        self.performance_history[op] = [
                            ModelPerformance(**entry) for entry in history
                        ]
                print("Loaded performance history")
            except Exception as e:
                print(f"Could not load performance history: {e}")
    
    def _save_performance_history(self):
        """Save performance history to disk"""
        history_file = "saved_models/performance_history.json"
        try:
            data = {}
            for op, history in self.performance_history.items():
                data[op] = [asdict(entry) for entry in history]
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Could not save performance history: {e}")
    
    def load_model_for_operation(self, operation: str, force_retrain: bool = False) -> Tuple[Any, str]:
        """
        Load the best available model for an operation
        Returns: (model, model_type)
        """
        if operation not in self.supported_operations:
            raise ValueError(f"Operation {operation} not supported")
        
        # Check cache first
        if not force_retrain and operation in self.models:
            best_type = self._get_best_model_type(operation)
            if best_type in self.models[operation]:
                return self.models[operation][best_type], best_type
        
        # Initialize models dict for this operation
        if operation not in self.models:
            self.models[operation] = {}
        
        # Determine which model to use based on history
        should_use_genetic = self._should_use_genetic_model(operation)
        
        if should_use_genetic:
            try:
                model, stats = train_genetic_model_advanced(
                    operation, 
                    generations=50, 
                    population_size=30
                )
                model_type = "genetic"
                
                # Benchmark the model
                benchmark = benchmark_genetic_model(model, operation)
                performance = ModelPerformance(
                    accuracy=stats['accuracy'],
                    avg_inference_time=benchmark['avg_time'],
                    training_time=stats.get('training_time', 0),
                    model_size=self._estimate_model_size(model),
                    generation=stats.get('generations', 0),
                    fitness_score=stats.get('best_fitness', 0)
                )
                
            except Exception as e:
                print(f"Genetic model training failed for {operation}: {e}")
                # Fallback to standard model
                should_use_genetic = False
        
        if not should_use_genetic:
            model, stats = train_or_load_model(
                operation, 
                epochs=1000, 
                force_retrain=force_retrain
            )
            model_type = "standard"
            
            # Benchmark the model
            benchmark = benchmark_neural_operation(model, operation)
            performance = ModelPerformance(
                accuracy=stats['accuracy'],
                avg_inference_time=benchmark['avg_time'],
                training_time=stats.get('training_time', 0),
                model_size=self._estimate_model_size(model)
            )
        
        # Cache the model
        if self.cache_models:
            self.models[operation][model_type] = model
        
        # Update performance history
        self.performance_history[operation].append(performance)
        self._save_performance_history()
        
        return model, model_type
    
    def execute_operation(self, operation: str, a: int, b: int) -> int:
        """Execute a single ternary operation using the best available model"""
        if operation not in self.supported_operations:
            raise ValueError(f"Operation {operation} not supported")
        
        # Update operation counter
        self.operation_counts[operation] += 1
        
        # Load model if needed
        model, model_type = self.load_model_for_operation(operation)
        
        # Prepare input
        input_data = np.array([[a, b]], dtype=np.float32)
        
        # Execute based on model type
        if model_type == "genetic":
            prediction = np.argmax(model.forward(input_data)[0])
        else:
            prediction = model.predict(input_data)[0]
        
        # Validate result is in ternary range
        result = int(np.clip(prediction, 0, 2))
        
        return result
    
    def execute_batch_operations(self, operation: str, operands: List[Tuple[int, int]]) -> List[int]:
        """Execute multiple operations in batch for efficiency"""
        if operation not in self.supported_operations:
            raise ValueError(f"Operation {operation} not supported")
        
        if not operands:
            return []
        
        # Update operation counter
        self.operation_counts[operation] += len(operands)
        
        # Load model if needed
        model, model_type = self.load_model_for_operation(operation)
        
        # Prepare input
        input_data = np.array(operands, dtype=np.float32)
        
        # Execute based on model type
        if model_type == "genetic":
            predictions = model.forward(input_data)
            results = np.argmax(predictions, axis=1)
        else:
            results = model.predict(input_data)
        
        # Validate results are in ternary range
        results = np.clip(results, 0, 2).astype(int)
        
        return results.tolist()
    
    def _should_use_genetic_model(self, operation: str) -> bool:
        """Decide whether to use genetic model based on historical performance"""
        if operation not in self.performance_history:
            return False
        
        history = self.performance_history[operation]
        if len(history) < 2:
            return False
        
        # Get recent performance data
        recent_genetic = [p for p in history[-10:] if p.generation > 0]
        recent_standard = [p for p in history[-10:] if p.generation == 0]
        
        if not recent_genetic or not recent_standard:
            return len(recent_genetic) > 0  # Use genetic if we only have genetic models
        
        # Compare average accuracies
        avg_genetic_accuracy = np.mean([p.accuracy for p in recent_genetic])
        avg_standard_accuracy = np.mean([p.accuracy for p in recent_standard])
        
        # Use genetic if it's significantly better or if standard model is below threshold
        return (avg_genetic_accuracy > avg_standard_accuracy + 0.02 or 
                avg_standard_accuracy < self.accuracy_threshold)
    
    def _get_best_model_type(self, operation: str) -> str:
        """Get the best model type for an operation based on cached performance"""
        if operation not in self.models or not self.models[operation]:
            return "standard"
        
        available_types = list(self.models[operation].keys())
        if len(available_types) == 1:
            return available_types[0]
        
        # If we have both, check recent performance
        if operation in self.performance_history:
            history = self.performance_history[operation]
            if history:
                latest = history[-1]
                return "genetic" if latest.generation > 0 else "standard"
        
        return "standard"  # Default fallback
    
    def _estimate_model_size(self, model) -> int:
        """Estimate model size in bytes"""
        total_params = 0
        
        if hasattr(model, 'weights') and hasattr(model, 'biases'):
            # Count parameters
            if isinstance(model.weights, dict):
                for w in model.weights.values():
                    total_params += w.size
            if isinstance(model.biases, dict):
                for b in model.biases.values():
                    total_params += b.size
        
        # Assume float32 (4 bytes per parameter)
        return total_params * 4
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'supported_operations': self.supported_operations,
            'loaded_models': {op: list(models.keys()) for op, models in self.models.items()},
            'operation_counts': self.operation_counts.copy(),
            'total_operations': sum(self.operation_counts.values()),
            'performance_summary': {}
        }
        
        # Add performance summary
        for op, history in self.performance_history.items():
            if history:
                latest = history[-1]
                status['performance_summary'][op] = {
                    'latest_accuracy': latest.accuracy,
                    'latest_inference_time': latest.avg_inference_time,
                    'model_type': 'genetic' if latest.generation > 0 else 'standard',
                    'total_evaluations': len(history)
                }
        
        return status
    
    def optimize_models(self, target_operations: Optional[List[str]] = None):
        """Optimize models for better performance"""
        if target_operations is None:
            # Optimize most used operations
            sorted_ops = sorted(self.operation_counts.items(), key=lambda x: x[1], reverse=True)
            target_operations = [op for op, count in sorted_ops[:5] if count > 10]
        
        print(f"Optimizing models for: {target_operations}")
        
        for operation in target_operations:
            if operation not in self.supported_operations:
                continue
            
            print(f"Optimizing {operation}...")
            
            # Get current best performance
            current_best = 0.0
            if operation in self.performance_history and self.performance_history[operation]:
                current_best = max(p.accuracy for p in self.performance_history[operation])
            
            # Try genetic optimization if current performance is below threshold
            if current_best < self.accuracy_threshold:
                try:
                    model, stats = train_genetic_model_advanced(
                        operation,
                        generations=100,
                        population_size=50
                    )
                    
                    if stats['accuracy'] > current_best:
                        print(f"Improved {operation}: {current_best:.4f} -> {stats['accuracy']:.4f}")
                        # Model will be automatically saved by train_genetic_model_advanced
                        
                        # Update cache
                        if self.cache_models:
                            if operation not in self.models:
                                self.models[operation] = {}
                            self.models[operation]['genetic'] = model
                    
                except Exception as e:
                    print(f"Optimization failed for {operation}: {e}")
    
    def benchmark_all_operations(self) -> Dict[str, Dict[str, Any]]:
        """Benchmark all supported operations"""
        results = {}
        
        for operation in self.supported_operations:
            try:
                model, model_type = self.load_model_for_operation(operation)
                
                if model_type == "genetic":
                    benchmark = benchmark_genetic_model(model, operation)
                else:
                    benchmark = benchmark_neural_operation(model, operation)
                
                results[operation] = {
                    'model_type': model_type,
                    'accuracy': benchmark['accuracy'],
                    'avg_time_ms': benchmark['avg_time'] * 1000,
                    'total_tests': benchmark['total_tests']
                }
                
            except Exception as e:
                results[operation] = {'error': str(e)}
        
        return results
    
    def cleanup(self):
        """Cleanup resources and save state"""
        print("Cleaning up TernaryNeuralCPU...")
        self._save_performance_history()
        
        # Clear model cache if needed
        if not self.cache_models:
            self.models.clear()
        
        print("Cleanup completed")


# Factory function for easy instantiation
def create_neural_cpu(auto_optimize: bool = True, cache_models: bool = True) -> TernaryNeuralCPU:
    """Create and initialize a TernaryNeuralCPU instance"""
    return TernaryNeuralCPU(auto_optimize=auto_optimize, cache_models=cache_models)
