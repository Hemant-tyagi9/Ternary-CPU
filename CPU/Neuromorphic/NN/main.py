#!/usr/bin/env python3
"""
TrinCore Neural Network - Main Execution Script
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import all necessary components
from TernaryNeuralCPU_nn import TernaryNeuralCPU, create_neural_cpu
from model_training import train_or_load_model, train_genetic_model_advanced, compare_models
from train_save_load import save_best_model, load_model, load_genetic_model
from genetic_evolution import evolve_all_operations
from nn_integration import run_integration_tests

def run_comprehensive_demo():
    """Run a comprehensive demonstration of all system capabilities"""
    print("\n" + "="*80)
    print("🚀 TrinCore Neural Network - Comprehensive Demonstration")
    print("="*80)
    
    # Initialize neural CPU
    neural_cpu = create_neural_cpu(auto_optimize=True, cache_models=True)
    
    # 1. Basic operations demonstration
    print("\n🔹 Basic Operations Demonstration")
    test_cases = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    operations = ["AND", "OR", "XOR", "NAND", "NOR", "ADD", "SUB"]
    
    for op in operations:
        print(f"\n{op} Results:")
        for a, b in test_cases:
            result = neural_cpu.execute_operation(op, a, b)
            print(f"  {op}({a}, {b}) = {result}")
    
    # 2. Batch processing demonstration
    print("\n🔹 Batch Processing Demonstration")
    batch_size = 100
    operands = [(np.random.randint(0, 3), np.random.randint(0, 3)) for _ in range(batch_size)]
    
    for op in ["AND", "OR", "XOR"]:
        start_time = time.time()
        results = neural_cpu.execute_batch_operations(op, operands)
        batch_time = time.time() - start_time
        print(f"  {op} batch processed {batch_size} operations in {batch_time*1000:.2f} ms")
    
    # 3. System status
    print("\n🔹 System Status")
    status = neural_cpu.get_system_status()
    print(f"Total operations executed: {status['total_operations']}")
    print("Operation counts:")
    for op, count in status['operation_counts'].items():
        print(f"  {op}: {count}")
    
    # 4. Benchmarking
    print("\n🔹 Benchmarking")
    benchmark_results = neural_cpu.benchmark_all_operations()
    print("\nBenchmark Results:")
    print(f"{'Operation':<8} {'Type':<10} {'Accuracy':<10} {'Time (ms)':<10}")
    print("-" * 40)
    for op, res in benchmark_results.items():
        if 'error' not in res:
            print(f"{op:<8} {res['model_type']:<10} {res['accuracy']:<10.4f} {res['avg_time_ms']:<10.3f}")
    
    # 5. Model training and comparison
    print("\n🔹 Model Training and Comparison")
    operations_to_train = ["AND", "OR", "XOR"]
    for op in operations_to_train:
        print(f"\nTraining models for {op}:")
        
        # Standard NN
        std_model, std_metrics = train_or_load_model(op)
        print(f"  Standard NN: {std_metrics['accuracy']:.4f} accuracy")
        
        # Genetic NN
        gen_model, gen_metrics = train_genetic_model_advanced(op, generations=100)
        print(f"  Genetic NN: {gen_metrics['accuracy']:.4f} accuracy")
        
        # Save best model
        if std_metrics['accuracy'] > gen_metrics['accuracy']:
            save_best_model(std_model, op, std_metrics, "standard")
        else:
            save_best_model(gen_model, op, gen_metrics, "genetic")
    
    # 6. Genetic evolution
    print("\n🔹 Genetic Evolution")
    evolve_all_operations()
    
    # 7. Integration tests
    print("\n🔹 Integration Tests")
    integration_success = run_integration_tests()
    
    print("\n" + "="*80)
    print("🏆 Demonstration Complete!")
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print("="*80)

def main_menu():
    """Display main menu and handle user input"""
    while True:
        print("\n" + "="*80)
        print("TrinCore Neural Network - Main Menu")
        print("="*80)
        print("1. Run Comprehensive Demonstration")
        print("2. Train Models for Specific Operation")
        print("3. Run Genetic Evolution")
        print("4. Run Integration Tests")
        print("5. Benchmark System Performance")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            run_comprehensive_demo()
        elif choice == "2":
            op = input("Enter operation to train (AND/OR/XOR/NAND/NOR/ADD/SUB): ").upper()
            if op in ["AND", "OR", "XOR", "NAND", "NOR", "ADD", "SUB"]:
                compare_models(op)
            else:
                print("Invalid operation!")
        elif choice == "3":
            evolve_all_operations()
        elif choice == "4":
            run_integration_tests()
        elif choice == "5":
            neural_cpu = create_neural_cpu()
            results = neural_cpu.benchmark_all_operations()
            print("\nBenchmark Results:")
            print(f"{'Operation':<8} {'Type':<10} {'Accuracy':<10} {'Time (ms)':<10}")
            print("-" * 40)
            for op, res in results.items():
                if 'error' not in res:
                    print(f"{op:<8} {res['model_type']:<10} {res['accuracy']:<10.4f} {res['avg_time_ms']:<10.3f}")
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    try:
        # Create necessary directories
        Path("models").mkdir(exist_ok=True)
        Path("results/plots").mkdir(parents=True, exist_ok=True)
        Path("saved_models").mkdir(exist_ok=True)
        
        # Display main menu
        main_menu()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
