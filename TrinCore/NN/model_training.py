import numpy as np
from .nn import TernaryGateNN, GeneticNeuralModel
from typing import Tuple, Dict, Any, List
import os
import sys
import time
from .train_save_load import save_best_model, load_model, load_genetic_model, get_model_filepath

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Cpu_components import ternary_gates

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
                result = ternary_gates.ternary_and(a, b)
            elif operation == "OR":
                result = ternary_gates.ternary_or(a, b)
            elif operation == "XOR":
                result = ternary_gates.ternary_xor(a, b)
            elif operation == "NAND":
                result = ternary_gates.ternary_nand(a, b)
            elif operation == "NOR":
                result = ternary_gates.ternary_nor(a, b)
            else:
                result = 0
            
            Y.append(result)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    Y_onehot = np.eye(3)[Y]
    
    return X, Y_onehot

def train_or_load_model(operation: str, input_size: int = 2, 
                       output_size: int = 3, hidden_layers: List[int] = None,
                       learning_rate: float = 0.01, epochs: int = 1000,
                       force_retrain: bool = False) -> Tuple[TernaryGateNN, Dict[str, Any]]:
    """Train a new model or load existing best model"""
    if hidden_layers is None:
        hidden_layers = [16, 12, 8]
    
    model_path = get_model_filepath(operation, "standard")
    
    # Try to load existing model first
    if not force_retrain and os.path.exists(model_path):
        try:
            model = load_model(model_path, input_size, hidden_layers, output_size, learning_rate)
            # Test the loaded model
            X, Y = generate_training_data(operation)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == np.argmax(Y, axis=1))
            
            print(f"Loaded existing model for {operation} with accuracy: {accuracy:.4f}")
            return model, {
                'operation': operation,
                'accuracy': accuracy,
                'training_loss': 0.0,  # Not available for loaded models
                'best_loss': 0.0,
                'epochs': 0,
                'loaded_from_file': True
            }
        except Exception as e:
            print(f"Failed to load existing model: {e}. Training new model...")
    
    # Train new model
    return train_model_for_operation(operation, input_size, output_size, 
                                   hidden_layers, learning_rate, epochs)

def train_model_for_operation(operation: str, input_size: int = 2, 
                            output_size: int = 3, hidden_layers: List[int] = None,
                            learning_rate: float = 0.01, epochs: int = 1000) -> Tuple[TernaryGateNN, Dict[str, Any]]:
    """Train a standard neural network for a ternary operation"""
    if hidden_layers is None:
        hidden_layers = [16, 12, 8]
    
    # Generate training data
    X, Y = generate_training_data(operation)
    
    model = TernaryGateNN(
        input_neurons=input_size,
        hidden1=hidden_layers[0],
        hidden2=hidden_layers[1],
        hidden3=hidden_layers[2],
        output_neurons=output_size,
        lr=learning_rate
    )
    
    # Train model
    print(f"Training standard NN for {operation}...")
    start_time = time.time()
    training_result = model.train(X, Y, epochs=epochs, patience=50, verbose=False)
    training_time = time.time() - start_time
    
    # Evaluate performance
    predictions = model.predict(X)
    accuracy = np.mean(predictions == np.argmax(Y, axis=1))
    
    metrics = {
        'operation': operation,
        'accuracy': accuracy,
        'training_loss': training_result['final_total_loss'],
        'best_loss': training_result['best_loss'],
        'epochs': training_result['total_epochs'],
        'training_time': training_time,
        'model_type': 'standard',
        'loaded_from_file': False
    }
    
    # Save if it's the best model
    save_best_model(model, operation, metrics, "standard")
    
    return model, metrics

def train_genetic_model_advanced(operation: str, population_size: int = 50, 
                                elite_size: int = 10, generations: int = 100,
                                mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                                architecture: List[int] = None) -> Tuple[GeneticNeuralModel, Dict[str, Any]]:
    """Enhanced genetic algorithm training with better evolution strategies"""
    
    if architecture is None:
        architecture = [2, 16, 12, 8, 3]
    
    # Try to load existing model first
    model_path = get_model_filepath(operation, "genetic")
    if os.path.exists(model_path):
        try:
            existing_model = load_genetic_model(model_path)
            # Test the loaded model
            X, Y = generate_training_data(operation)
            y_true = np.argmax(Y, axis=1)
            predictions = existing_model.forward(X)
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y_true)
            
            if accuracy > 0.8:  # If existing model is good enough
                print(f"Loaded existing genetic model for {operation} with accuracy: {accuracy:.4f}")
                return existing_model, {
                    'operation': operation,
                    'accuracy': accuracy,
                    'best_fitness': existing_model.fitness,
                    'generations': existing_model.generation,
                    'model_type': 'genetic',
                    'loaded_from_file': True
                }
        except Exception as e:
            print(f"Failed to load existing genetic model: {e}. Training new model...")
    
    # Generate training data
    X, Y = generate_training_data(operation)
    y_true = np.argmax(Y, axis=1)
    
    print(f"Training genetic NN for {operation}...")
    start_time = time.time()
    
    # Initialize population with diversity
    population = []
    for i in range(population_size):
        model = GeneticNeuralModel(architecture, mutation_rate)
        # Add some diversity to initial population
        if i > 0:
            model = model.mutate()
        population.append(model)
    
    best_fitness_history = []
    avg_fitness_history = []
    best_accuracy_history = []
    
    for generation in range(generations):
        # Evaluate fitness with improved scoring
        fitnesses = []
        accuracies = []
        
        for model in population:
            predictions = model.forward(X)
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y_true)
            
            # Improved fitness function
            entropy_loss = -np.mean(np.sum(Y * np.log(predictions + 1e-8), axis=1))
            diversity_bonus = np.std(predictions.flatten()) * 10  # Reward diverse outputs
            fitness = accuracy * 1000 - entropy_loss * 50 + diversity_bonus
            
            model.fitness = fitness
            model.generation = generation
            fitnesses.append(fitness)
            accuracies.append(accuracy)
        
        # Track progress
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        best_accuracy = max(accuracies)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        best_accuracy_history.append(best_accuracy)
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Accuracy = {best_accuracy:.4f}, "
                  f"Avg Fitness = {avg_fitness:.2f}")
        
        # Early stopping if we achieve perfect accuracy
        if best_accuracy >= 0.99:
            print(f"Perfect accuracy achieved at generation {generation}")
            break
        
        # Selection with tournament selection
        sorted_indices = np.argsort(fitnesses)[::-1]
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # Create new generation with improved strategies
        new_population = elite.copy()
        
        while len(new_population) < population_size:
            if np.random.random() < crossover_rate:
                # Tournament selection for parents
                parent1 = tournament_selection(population, fitnesses, 3)
                parent2 = tournament_selection(population, fitnesses, 3)
                child = parent1.crossover(parent2)
            else:
                # Select from elite for mutation
                parent = elite[np.random.randint(0, len(elite))]
                child = GeneticNeuralModel(parent.architecture, parent.mutation_rate)
                child.weights = {k: v.copy() for k, v in parent.weights.items()}
                child.biases = {k: v.copy() for k, v in parent.biases.items()}
            
            # Adaptive mutation rate
            adaptive_mutation_rate = mutation_rate * (1 + 0.5 * (generations - generation) / generations)
            child.mutation_rate = adaptive_mutation_rate
            child = child.mutate()
            
            new_population.append(child)
        
        population = new_population
    
    training_time = time.time() - start_time
    
    # Get the best model
    final_fitnesses = [model.fitness for model in population]
    best_idx = np.argmax(final_fitnesses)
    best_model = population[best_idx]
    
    # Final evaluation
    predictions = best_model.forward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    final_accuracy = np.mean(predicted_classes == y_true)
    
    metrics = {
        'operation': operation,
        'accuracy': final_accuracy,
        'best_fitness': best_model.fitness,
        'generations': generation + 1,
        'training_time': training_time,
        'model_type': 'genetic',
        'fitness_history': best_fitness_history,
        'accuracy_history': best_accuracy_history,
        'loaded_from_file': False
    }
    
    # Save if it's the best model
    save_best_model(best_model, operation, metrics, "genetic")
    
    return best_model, metrics

def tournament_selection(population: List[GeneticNeuralModel], 
                        fitnesses: List[float], tournament_size: int = 3) -> GeneticNeuralModel:
    """Select a parent using tournament selection"""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

# Legacy function for backward compatibility
def train_genetic_model(operation: str, population_size: int = 50, 
                       elite_size: int = 10, generations: int = 100) -> GeneticNeuralModel:
    """Train a genetic neural network for a specific ternary operation (legacy)"""
    model, _ = train_genetic_model_advanced(operation, population_size, elite_size, generations)
    return model

def compare_models(operation: str, force_retrain: bool = False) -> Dict[str, Any]:
    """Compare standard NN vs genetic NN for an operation"""
    print(f"\n=== Comparing Models for {operation} ===")
    
    # Train/load standard model
    std_model, std_metrics = train_or_load_model(operation, force_retrain=force_retrain)
    
    # Train/load genetic model
    gen_model, gen_metrics = train_genetic_model_advanced(operation, generations=50)
    
    comparison = {
        'operation': operation,
        'standard': std_metrics,
        'genetic': gen_metrics,
        'winner': 'standard' if std_metrics['accuracy'] > gen_metrics['accuracy'] else 'genetic'
    }
    
    print(f"Standard NN: {std_metrics['accuracy']:.4f} accuracy")
    print(f"Genetic NN: {gen_metrics['accuracy']:.4f} accuracy")
    print(f"Winner: {comparison['winner']}")
    
    return comparison
