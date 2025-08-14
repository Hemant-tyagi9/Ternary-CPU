#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm for Ternary Neural Networks - FIXED VERSION
Fixed dimension issues and optimized for 100% accuracy achievement
"""

import numpy as np
import time
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Ternary operations with explicit type checking
def ternary_and(a: int, b: int) -> int:
    """Ternary AND operation with proper bounds checking"""
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

@dataclass
class TrainingConfig:
    """Configuration for genetic training"""
    population_size: int = 200
    generations: int = 300
    elite_ratio: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.8
    tournament_size: int = 5
    hidden_size: int = 32
    early_stop_accuracy: float = 1.0
    patience: int = 50
    adaptive_mutation: bool = True

def prepare_dataset(operation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare complete training dataset with validation"""
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")
    
    op_func = OPERATIONS[operation]
    
    # Generate all possible combinations
    X, Y, Y_onehot = [], [], []
    
    for a in range(3):
        for b in range(3):
            X.append([a, b])
            result = op_func(a, b)
            Y.append(result)
            
            # One-hot encoding
            y_vec = np.zeros(3, dtype=np.float32)
            y_vec[result] = 1.0
            Y_onehot.append(y_vec)
    
    return (np.array(X, dtype=np.float32), 
            np.array(Y, dtype=np.int32), 
            np.array(Y_onehot, dtype=np.float32))

class EnhancedGeneticModel:
    """Enhanced genetic model with fixed architecture and improved stability"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize with proper Xavier/He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        
        self.fitness = 0.0
        self.accuracy = 0.0
        self.generation = 0
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with numerical stability"""
        # Ensure input is properly shaped
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Hidden layer with ReLU activation
        z1 = np.dot(X, self.W1) + self.b1
        h1 = np.maximum(0, z1)  # ReLU
        
        # Output layer
        z2 = np.dot(h1, self.W2) + self.b2
        
        # Stable softmax
        z2_max = np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(np.clip(z2 - z2_max, -500, 500))
        return exp_z2 / (np.sum(exp_z2, axis=1, keepdims=True) + 1e-8)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray, Y_onehot: np.ndarray) -> Tuple[float, float]:
        """Comprehensive evaluation with multiple metrics"""
        try:
            # Get predictions
            probs = self.forward(X)
            predictions = np.argmax(probs, axis=1)
            
            # Accuracy
            accuracy = np.mean(predictions == Y)
            
            # Cross-entropy loss
            epsilon = 1e-8
            ce_loss = -np.mean(np.sum(Y_onehot * np.log(probs + epsilon), axis=1))
            
            # Confidence measure
            correct_probs = probs[np.arange(len(Y)), Y]
            avg_confidence = np.mean(correct_probs)
            
            # Combined fitness
            fitness = (accuracy ** 3) * avg_confidence * 100 - ce_loss * 10
            
            # Bonus for perfect accuracy
            if accuracy == 1.0:
                fitness += 1000
                
            return accuracy, max(fitness, 0.0)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0, 0.0
    
    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 0.1) -> 'EnhancedGeneticModel':
        """Mutation with fixed architecture"""
        child = EnhancedGeneticModel(self.input_dim, self.hidden_dim, self.output_dim)
        
        # Copy weights with mutation
        child.W1 = self.W1.copy()
        child.b1 = self.b1.copy()
        child.W2 = self.W2.copy()
        child.b2 = self.b2.copy()
        
        # Apply mutation
        if np.random.random() < mutation_rate:
            mask = np.random.random(self.W1.shape) < mutation_rate
            noise = np.random.normal(0, mutation_strength, self.W1.shape).astype(np.float32)
            child.W1 += noise * mask
            
        if np.random.random() < mutation_rate:
            mask = np.random.random(self.b1.shape) < mutation_rate
            noise = np.random.normal(0, mutation_strength * 0.5, self.b1.shape).astype(np.float32)
            child.b1 += noise * mask
            
        if np.random.random() < mutation_rate:
            mask = np.random.random(self.W2.shape) < mutation_rate
            noise = np.random.normal(0, mutation_strength, self.W2.shape).astype(np.float32)
            child.W2 += noise * mask
            
        if np.random.random() < mutation_rate:
            mask = np.random.random(self.b2.shape) < mutation_rate
            noise = np.random.normal(0, mutation_strength * 0.5, self.b2.shape).astype(np.float32)
            child.b2 += noise * mask
        
        child.generation = self.generation + 1
        return child
    
    def crossover(self, partner: 'EnhancedGeneticModel') -> 'EnhancedGeneticModel':
        """Crossover with fixed architecture"""
        child = EnhancedGeneticModel(self.input_dim, self.hidden_dim, self.output_dim)
        child.generation = max(self.generation, partner.generation) + 1
        
        # Blend crossover for weights
        alpha = 0.3  # Blend factor
        mask = np.random.random(self.W1.shape) < 0.5
        child.W1 = np.where(mask, 
                          self.W1 * (1 + alpha) - partner.W1 * alpha,
                          partner.W1 * (1 + alpha) - self.W1 * alpha)
        
        # Average crossover for biases
        child.b1 = (self.b1 + partner.b1) / 2
        
        # Blend crossover for output weights
        mask = np.random.random(self.W2.shape) < 0.5
        child.W2 = np.where(mask,
                          self.W2 * (1 + alpha) - partner.W2 * alpha,
                          partner.W2 * (1 + alpha) - self.W2 * alpha)
        
        # Average crossover for output biases
        child.b2 = (self.b2 + partner.b2) / 2
        
        return child

def tournament_selection(population: List[EnhancedGeneticModel], 
                        tournament_size: int = 5) -> EnhancedGeneticModel:
    """Tournament selection with diversity consideration"""
    if len(population) < tournament_size:
        tournament_size = len(population)
    
    participants = np.random.choice(len(population), tournament_size, replace=False)
    participants = [population[i] for i in participants]
    
    return max(participants, key=lambda x: x.fitness)

def evolve_operation_advanced(operation: str, config: TrainingConfig) -> Tuple[Optional[EnhancedGeneticModel], Dict[str, List]]:
    """Advanced evolution with fixed architecture"""
    print(f"\nEVOLVING {operation} - TARGET: 100% ACCURACY")
    print(f"Population: {config.population_size}, Max Generations: {config.generations}")
    print("-" * 60)
    
    # Prepare dataset
    X, Y, Y_onehot = prepare_dataset(operation)
    print(f"Dataset: {len(X)} samples, {operation} operation")
    
    # Initialize population with fixed architecture
    population = []
    for _ in range(config.population_size):
        model = EnhancedGeneticModel(hidden_dim=config.hidden_size)
        population.append(model)
    
    # Evolution tracking
    history = {
        'best_accuracy': [],
        'avg_accuracy': [],
        'best_fitness': [],
        'avg_fitness': [],
        'perfect_count': []
    }
    
    best_model = None
    best_accuracy = 0.0
    perfect_models = []
    stagnation_counter = 0
    
    for generation in range(config.generations):
        # Evaluate population
        accuracies = []
        fitnesses = []
        
        for model in population:
            accuracy, fitness = model.evaluate(X, Y, Y_onehot)
            model.accuracy = accuracy
            model.fitness = fitness
            model.generation = generation
            
            accuracies.append(accuracy)
            fitnesses.append(fitness)
            
            if accuracy >= config.early_stop_accuracy:
                if model not in perfect_models:
                    perfect_models.append(model)
        
        # Statistics
        avg_accuracy = np.mean(accuracies)
        max_accuracy = np.max(accuracies)
        avg_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        perfect_count = len(perfect_models)
        
        # Update history
        history['best_accuracy'].append(max_accuracy)
        history['avg_accuracy'].append(avg_accuracy)
        history['best_fitness'].append(max_fitness)
        history['avg_fitness'].append(avg_fitness)
        history['perfect_count'].append(perfect_count)
        
        # Update best model
        current_best = max(population, key=lambda x: x.fitness)
        if max_accuracy > best_accuracy:
            best_accuracy = max_accuracy
            best_model = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Progress reporting
        if generation % 20 == 0 or generation == config.generations - 1:
            print(f"Gen {generation:3d}: Best={max_accuracy:.4f}, Avg={avg_accuracy:.4f}, "
                  f"Perfect={perfect_count}, Fitness={max_fitness:.2f}")
        
        # Early stopping if we have multiple perfect models
        if perfect_count >= 5:
            print(f"EARLY STOP: {perfect_count} perfect models found at generation {generation}")
            break
        
        # Create next generation
        elite_count = max(5, int(config.population_size * config.elite_ratio))
        population.sort(key=lambda x: x.fitness, reverse=True)
        elite = population[:elite_count]
        
        new_population = elite.copy()
        
        # Adaptive parameters
        progress = generation / config.generations
        current_mutation_rate = config.mutation_rate * (1.5 - progress)
        current_mutation_strength = 0.2 * (1.0 - progress * 0.7)
        
        while len(new_population) < config.population_size:
            if np.random.random() < config.crossover_rate and len(elite) >= 2:
                # Crossover
                parent1 = tournament_selection(population, config.tournament_size)
                parent2 = tournament_selection(population, config.tournament_size)
                child = parent1.crossover(parent2)
                
                # Apply light mutation after crossover
                if np.random.random() < 0.7:
                    child = child.mutate(current_mutation_rate * 0.5, current_mutation_strength * 0.5)
            else:
                # Mutation only
                parent = tournament_selection(population, config.tournament_size)
                child = parent.mutate(current_mutation_rate, current_mutation_strength)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best perfect model if available
    final_model = None
    if perfect_models:
        final_model = max(perfect_models, key=lambda x: x.fitness)
        print(f"PERFECT MODEL FOUND! Accuracy: {final_model.accuracy:.6f}")
    elif best_model:
        final_model = best_model
        print(f"BEST MODEL: Accuracy: {final_model.accuracy:.6f}")
    else:
        print("NO VALID MODEL PRODUCED")
    
    return final_model, history

def save_model_enhanced(model: EnhancedGeneticModel, operation: str, 
                       accuracy: float, history: Dict[str, List]):
    """Save model with comprehensive metadata"""
    models_dir = Path("models_genetic_enhanced")
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2,
        'accuracy': accuracy,
        'fitness': model.fitness,
        'generation': model.generation,
        'architecture': f"{model.input_dim}-{model.hidden_dim}-{model.output_dim}",
        'operation': operation
    }
    
    model_file = models_dir / f"{operation.lower()}_perfect_model.npz"
    np.savez_compressed(model_file, **model_data)
    
    history_file = models_dir / f"{operation.lower()}_training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved {operation} model: {model_file} (Accuracy: {accuracy:.6f})")

def test_model_thoroughly(model: EnhancedGeneticModel, operation: str) -> bool:
    """Thorough testing of the model"""
    X, Y, Y_onehot = prepare_dataset(operation)
    
    predictions = model.predict(X)
    accuracy = np.mean(predictions == Y)
    
    print(f"\nTHOROUGH TEST for {operation}:")
    print("Input | Expected | Predicted | Correct")
    print("-" * 40)
    
    all_correct = True
    for i, ((a, b), expected, predicted) in enumerate(zip(X, Y, predictions)):
        correct = "✓" if expected == predicted else "✗"
        if expected != predicted:
            all_correct = False
        print(f"({int(a)},{int(b)}) |    {expected}     |     {predicted}     |   {correct}")
    
    print(f"\nFinal Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")
    return all_correct

def evolve_all_operations_enhanced():
    """Evolve all operations with enhanced genetic algorithm"""
    print("ENHANCED GENETIC EVOLUTION - TARGET: 100% ACCURACY")
    print("=" * 80)
    
    operations = ['AND', 'OR', 'XOR', 'ADD', 'SUB', 'NAND', 'NOR']
    
    # Enhanced configuration
    config = TrainingConfig(
        population_size=150,
        generations=500,
        elite_ratio=0.15,
        mutation_rate=0.4,
        crossover_rate=0.75,
        tournament_size=7,
        hidden_size=24,  # Fixed hidden size
        early_stop_accuracy=1.0,
        patience=75,
        adaptive_mutation=True
    )
    
    results = {}
    perfect_count = 0
    
    for operation in operations:
        print(f"\n{'='*25} {operation} {'='*25}")
        start_time = time.time()
        
        try:
            best_model, history = evolve_operation_advanced(operation, config)
            elapsed = time.time() - start_time
            
            if best_model is not None:
                is_perfect = test_model_thoroughly(best_model, operation)
                
                if is_perfect and best_model.accuracy >= 1.0:
                    perfect_count += 1
                    status = "PERFECT"
                    save_model_enhanced(best_model, operation, best_model.accuracy, history)
                else:
                    status = f"{best_model.accuracy:.4f}"
                
                results[operation] = {
                    'status': 'SUCCESS',
                    'accuracy': best_model.accuracy,
                    'perfect': is_perfect,
                    'time': elapsed,
                    'generations': best_model.generation
                }
                
                print(f"{operation}: {status} in {elapsed:.1f}s (Gen {best_model.generation})")
            else:
                results[operation] = {'status': 'FAILED', 'accuracy': 0.0, 'perfect': False}
                print(f"{operation}: FAILED")
                
        except Exception as e:
            results[operation] = {'status': 'ERROR', 'accuracy': 0.0, 'perfect': False}
            print(f"{operation}: ERROR - {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS - ENHANCED GENETIC EVOLUTION")
    print("=" * 80)
    
    total_time = sum(r.get('time', 0) for r in results.values() if 'time' in r)
    
    for operation, result in results.items():
        if result['status'] == 'SUCCESS':
            status_icon = "🎯" if result['perfect'] else "📊"
            acc_str = "100.0000%" if result['perfect'] else f"{result['accuracy']*100:.4f}%"
            time_str = f"{result.get('time', 0):.1f}s" if 'time' in result else "N/A"
            gen_str = f"G{result.get('generations', 0)}" if 'generations' in result else "N/A"
            print(f"{status_icon} {operation:4s}: {acc_str:>10s} | {time_str:>6s} | {gen_str:>4s}")
        else:
            print(f"{operation:4s}: {result['status']:>10s}")
    
    print(f"\nPERFECT MODELS: {perfect_count}/{len(operations)}")
    print(f"SUCCESS RATE: {len([r for r in results.values() if r['status'] == 'SUCCESS'])}/{len(operations)}")
    print(f"TOTAL TIME: {total_time:.1f}s")
    
    if perfect_count == len(operations):
        print("\nALL OPERATIONS ACHIEVED 100% ACCURACY!")
    else:
        print(f"\n{perfect_count} operations achieved 100% accuracy")

if __name__ == '__main__':
    evolve_all_operations_enhanced()
