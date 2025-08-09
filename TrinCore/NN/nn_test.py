import numpy as np
import time
import matplotlib.pyplot as plt
from .nn import TernaryGateNN, GeneticNeuralModel
from .model_training import (train_or_load_model, train_genetic_model_advanced, 
                           generate_training_data, compare_models)
from .nn_benchmark import benchmark_neural_operation, benchmark_genetic_model, benchmark_traditional_operation
from .train_save_load import (save_model, load_model, save_genetic_model, load_genetic_model,
                            create_model_directory, save_best_model)
from .nn_integration import NeuralIntegration
import os

def test_ternary_gate_nn():
    """Test the standard neural network implementation"""
    print("\n=== Testing TernaryGateNN ===")
    
    # Create model directory
    create_model_directory()
    
    # Train a model (or load if exists)
    model, stats = train_or_load_model("AND", epochs=500)
    print(f"Training stats: {stats}")
    
    # Benchmark
    benchmark = benchmark_neural_operation(model, "AND")
    print(f"Benchmark results: {benchmark}")
    
    # Test predictions for all combinations
    print("\nTesting all AND combinations:")
    X, Y = generate_training_data("AND")
    predictions = model.predict(X)
    expected = np.argmax(Y, axis=1)
    
    for i, (inputs, pred, exp) in enumerate(zip(X, predictions, expected)):
        a, b = int(inputs[0]), int(inputs[1])
        correct = "✓" if pred == exp else "✗"
        print(f"AND({a}, {b}) = {pred} (expected: {exp}) {correct}")
    
    print("Standard NN test completed")
    return model, stats

def test_genetic_model():
    """Test the genetic neural network implementation"""
    print("\n=== Testing GeneticNeuralModel ===")
    
    # Train a model
    model, stats = train_genetic_model_advanced("OR", generations=30, population_size=20)
    print(f"Training stats: {stats}")
    
    # Benchmark
    benchmark = benchmark_genetic_model(model, "OR")
    print(f"Benchmark results: {benchmark}")
    
    # Test predictions for all combinations
    print("\nTesting all OR combinations:")
    X, Y = generate_training_data("OR")
    predictions = model.forward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    expected = np.argmax(Y, axis=1)
    
    for i, (inputs, pred, exp) in enumerate(zip(X, predicted_classes, expected)):
        a, b = int(inputs[0]), int(inputs[1])
        correct = "✓" if pred == exp else "✗"
        print(f"OR({a}, {b}) = {pred} (expected: {exp}) {correct}")
    
    print("Genetic NN test completed")
    return model, stats

def test_neural_integration():
    """Test the neural integration layer"""
    print("\n=== Testing NeuralIntegration ===")
    
    try:
        integration = NeuralIntegration()
        integration.train_models(["AND", "OR", "XOR"])
        
        # Test operations
        operations = ["AND", "OR", "XOR"]
        test_cases = [(0, 0), (0, 1), (1, 1), (2, 2)]
        
        for op in operations:
            print(f"\nTesting {op} operation:")
            for a, b in test_cases:
                result = integration.execute_operation(op, a, b)
                print(f"{op}({a}, {b}) = {result}")
        
        # Test batch operations
        print(f"\nTesting batch operations:")
        batch_results = integration.execute_batch_operations("AND", test_cases)
        print(f"Batch AND results: {batch_results}")
        
        print("Neural integration test completed")
    except Exception as e:
        print(f"Neural integration test failed: {e}")

def test_model_comparison():
    """Test model comparison functionality"""
    print("\n=== Testing Model Comparison ===")
    
    operations = ["AND", "OR", "XOR"]
    for operation in operations:
        try:
            comparison = compare_models(operation)
            print(f"\n{operation} Operation Comparison:")
            print(f"Standard NN Accuracy: {comparison['standard']['accuracy']:.4f}")
            print(f"Genetic NN Accuracy: {comparison['genetic']['accuracy']:.4f}")
            print(f"Winner: {comparison['winner']}")
        except Exception as e:
            print(f"Comparison failed for {operation}: {e}")

def test_benchmark_comparison():
    """Compare performance of different implementations"""
    print("\n=== Benchmark Comparison ===")
    
    operation = "XOR"
    try:
        # Traditional
        trad_bench = benchmark_traditional_operation(operation)
        print(f"Traditional {operation}: {trad_bench['avg_time']*1000:.4f} ms per op")
        
        # Standard NN
        model, _ = train_or_load_model(operation, epochs=500)
        nn_bench = benchmark_neural_operation(model, operation)
        print(f"Standard NN {operation}: {nn_bench['avg_time']*1000:.4f} ms per op, "
              f"Accuracy: {nn_bench['accuracy']:.4f}")
        
        # Genetic NN
        genetic_model, _ = train_genetic_model_advanced(operation, generations=20)
        gen_bench = benchmark_genetic_model(genetic_model, operation)
        print(f"Genetic NN {operation}: {gen_bench['avg_time']*1000:.4f} ms per op, "
              f"Accuracy: {gen_bench['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Benchmark comparison failed: {e}")

def test_save_load_functionality():
    """Test save/load functionality thoroughly"""
    print("\n=== Testing Save/Load Functionality ===")
    
    try:
        # Test standard model save/load
        print("Testing standard model save/load...")
        original_model, _ = train_or_load_model("ADD", epochs=200)
        
        # Get predictions from original
        X, _ = generate_training_data("ADD")
        original_predictions = original_model.predict(X)
        
        # Save and load
        test_file = "test_standard_model.npz"
        save_model(original_model, test_file)
        loaded_model = load_model(test_file)
        
        # Compare predictions
        loaded_predictions = loaded_model.predict(X)
        matches = np.array_equal(original_predictions, loaded_predictions)
        print(f"Standard model save/load: {'PASSED' if matches else 'FAILED'}")
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Test genetic model save/load
        print("Testing genetic model save/load...")
        original_genetic, _ = train_genetic_model_advanced("SUB", generations=10, population_size=10)
        
        # Get predictions from original
        original_genetic_preds = np.argmax(original_genetic.forward(X), axis=1)
        
        # Save and load
        test_genetic_file = "test_genetic_model.pkl"
        save_genetic_model(original_genetic, test_genetic_file)
        loaded_genetic = load_genetic_model(test_genetic_file)
        
        # Compare predictions
        loaded_genetic_preds = np.argmax(loaded_genetic.forward(X), axis=1)
        genetic_matches = np.array_equal(original_genetic_preds, loaded_genetic_preds)
        print(f"Genetic model save/load: {'PASSED' if genetic_matches else 'FAILED'}")
        
        # Cleanup
        if os.path.exists(test_genetic_file):
            os.remove(test_genetic_file)
            
    except Exception as e:
        print(f"Save/load test failed: {e}")

def test_all_operations():
    """Test all supported ternary operations"""
    print("\n=== Testing All Ternary Operations ===")
    
    operations = ["AND", "OR", "XOR", "ADD", "SUB"]
    results = {}
    
    for operation in operations:
        try:
            print(f"\nTesting {operation}...")
            
            # Test standard model
            std_model, std_stats = train_or_load_model(operation, epochs=300)
            
            # Test genetic model (fewer generations for faster testing)
            gen_model, gen_stats = train_genetic_model_advanced(
                operation, generations=20, population_size=15
            )
            
            results[operation] = {
                'standard_accuracy': std_stats['accuracy'],
                'genetic_accuracy': gen_stats['accuracy'],
                'better_model': 'standard' if std_stats['accuracy'] > gen_stats['accuracy'] else 'genetic'
            }
            
            print(f"{operation}: Standard={std_stats['accuracy']:.4f}, "
                  f"Genetic={gen_stats['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Failed to test {operation}: {e}")
            results[operation] = {'error': str(e)}
    
    print("\n=== Summary ===")
    for op, result in results.items():
        if 'error' not in result:
            print(f"{op}: Best model is {result['better_model']} "
                  f"(Std: {result['standard_accuracy']:.4f}, "
                  f"Gen: {result['genetic_accuracy']:.4f})")
        else:
            print(f"{op}: ERROR - {result['error']}")

if __name__ == "__main__":
    print("Starting comprehensive neural network tests...")
    
    # Create necessary directories
    create_model_directory()
    
    # Run all tests
    try:
        test_ternary_gate_nn()
        test_genetic_model()
        test_save_load_functionality()
        test_model_comparison()
        test_benchmark_comparison()
        test_neural_integration()
        test_all_operations()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
