#!/usr/bin/env python3

import sys
import os
import numpy as np
import time

# Add the NN module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_basic_operations():
    """Demonstrate basic neural CPU operations"""
    print("=" * 60)
    print("BASIC OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Test all supported operations
    test_cases = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    operations = ["AND", "OR", "XOR", "ADD", "SUB"]
    
    for operation in operations:
        print(f"\n{operation} Operation Results:")
        print("-" * 30)
        for a, b in test_cases:
            try:
                result = neural_cpu.execute_operation(operation, a, b)
                print(f"{operation}({a}, {b}) = {result}")
            except Exception as e:
                print(f"{operation}({a}, {b}) = ERROR: {e}")
    
    return neural_cpu

def demonstrate_batch_processing(neural_cpu):
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create large batch of operations
    batch_size = 100
    operands = [(np.random.randint(0, 3), np.random.randint(0, 3)) for _ in range(batch_size)]
    
    # Time batch vs individual operations
    operations = ["AND", "OR", "XOR"]
    
    for operation in operations:
        print(f"\nTesting {operation} - Batch vs Individual:")
        print("-" * 40)
        
        # Batch processing
        start_time = time.time()
        batch_results = neural_cpu.execute_batch_operations(operation, operands)
        batch_time = time.time() - start_time
        
        # Individual processing
        start_time = time.time()
        individual_results = []
        for a, b in operands:
            individual_results.append(neural_cpu.execute_operation(operation, a, b))
        individual_time = time.time() - start_time
        
        # Verify results match
        results_match = batch_results == individual_results
        
        print(f"Batch time: {batch_time*1000:.2f} ms ({batch_time*1000/batch_size:.3f} ms per op)")
        print(f"Individual time: {individual_time*1000:.2f} ms ({individual_time*1000/batch_size:.3f} ms per op)")
        print(f"Speedup: {individual_time/batch_time:.2f}x")
        print(f"Results match: {results_match}")

def demonstrate_system_status(neural_cpu):
    """Demonstrate system status and monitoring"""
    print("\n" + "=" * 60)
    print("SYSTEM STATUS DEMONSTRATION")
    print("=" * 60)
    
    status = neural_cpu.get_system_status()
    
    print(f"Supported Operations: {', '.join(status['supported_operations'])}")
    print(f"Total Operations Executed: {status['total_operations']}")
    
    print("\nOperation Counts:")
    for op, count in status['operation_counts'].items():
        print(f"  {op}: {count} times")
    
    print("\nLoaded Models:")
    for op, model_types in status['loaded_models'].items():
        if model_types:
            print(f"  {op}: {', '.join(model_types)}")
    
    if status['performance_summary']:
        print("\nPerformance Summary:")
        for op, perf in status['performance_summary'].items():
            print(f"  {op}: {perf['latest_accuracy']:.4f} accuracy, "
                  f"{perf['latest_inference_time']*1000:.3f} ms, "
                  f"{perf['model_type']} model")

def demonstrate_benchmarking(neural_cpu):
    """Demonstrate comprehensive benchmarking"""
    print("\n" + "=" * 60)
    print("BENCHMARKING DEMONSTRATION")
    print("=" * 60)
    
    print("Running comprehensive benchmark...")
    benchmark_results = neural_cpu.benchmark_all_operations()
    
    print(f"\n{'Operation':<8} {'Model':<8} {'Accuracy':<10} {'Time (ms)':<12} {'Status'}")
    print("-" * 50)
    
    for operation, results in benchmark_results.items():
        if 'error' in results:
            print(f"{operation:<8} {'ERROR':<8} {'N/A':<10} {'N/A':<12} {results['error']}")
        else:
            print(f"{operation:<8} {results['model_type']:<8} "
                  f"{results['accuracy']:<10.4f} {results['avg_time_ms']:<12.3f} "
                  f"{'PASS' if results['accuracy'] > 0.8 else 'FAIL'}")

def demonstrate_optimization(neural_cpu):
    """Demonstrate model optimization"""
    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Get initial performance
    print("Initial performance:")
    initial_results = neural_cpu.benchmark_all_operations()
    
    for op, results in initial_results.items():
        if 'accuracy' in results:
            print(f"  {op}: {results['accuracy']:.4f} accuracy")
    
    # Run optimization on operations with low accuracy
    operations_to_optimize = []
    for op, results in initial_results.items():
        if 'accuracy' in results and results['accuracy'] < 0.9:
            operations_to_optimize.append(op)
    
    if operations_to_optimize:
        print(f"\nOptimizing operations: {', '.join(operations_to_optimize)}")
        neural_cpu.optimize_models(operations_to_optimize)
        
        # Check improved performance
        print("\nPost-optimization performance:")
        final_results = neural_cpu.benchmark_all_operations()
        
        for op in operations_to_optimize:
            if op in final_results and 'accuracy' in final_results[op]:
                initial_acc = initial_results[op].get('accuracy', 0)
                final_acc = final_results[op]['accuracy']
                improvement = final_acc - initial_acc
                print(f"  {op}: {initial_acc:.4f} -> {final_acc:.4f} "
                      f"({improvement:+.4f})")
    else:
        print("All models already performing well (>90% accuracy)")

def demonstrate_real_world_scenario(neural_cpu):
    """Demonstrate a real-world scenario"""
    print("\n" + "=" * 60)
    print("REAL-WORLD SCENARIO: TERNARY LOGIC CIRCUIT SIMULATION")
    print("=" * 60)
    
    print("Simulating a complex ternary logic circuit...")
    print("Circuit: (A AND B) OR (C XOR D) AND (E ADD F)")
    
    # Generate random inputs
    num_tests = 50
    circuit_results = []
    
    for i in range(num_tests):
        # Generate random ternary inputs
        A, B, C, D, E, F = [np.random.randint(0, 3) for _ in range(6)]
        
        # Simulate the circuit using neural CPU
        start_time = time.time()
        
        # Stage 1: Parallel operations
        stage1_ops = [("AND", A, B), ("XOR", C, D), ("ADD", E, F)]
        stage1_operands = [(op[1], op[2]) for op in stage1_ops]
        
        and_result = neural_cpu.execute_operation("AND", A, B)
        xor_result = neural_cpu.execute_operation("XOR", C, D)
        add_result = neural_cpu.execute_operation("ADD", E, F)
        
        # Stage 2: Combine results
        temp_result = neural_cpu.execute_operation("OR", and_result, xor_result)
        final_result = neural_cpu.execute_operation("AND", temp_result, add_result)
        
        execution_time = time.time() - start_time
        
        circuit_results.append({
            'inputs': (A, B, C, D, E, F),
            'intermediate': (and_result, xor_result, add_result, temp_result),
            'output': final_result,
            'time': execution_time
        })
        
        if i < 5:  # Show first 5 results
            print(f"Test {i+1}: Inputs({A},{B},{C},{D},{E},{F}) -> Output({final_result}) "
                  f"[{execution_time*1000:.3f}ms]")
    
    # Statistics
    avg_time = np.mean([r['time'] for r in circuit_results])
    total_ops = len(circuit_results) * 5  # 5 operations per circuit
    
    print(f"\nCircuit Simulation Results:")
    print(f"  Tests run: {num_tests}")
    print(f"  Total operations: {total_ops}")
    print(f"  Average time per circuit: {avg_time*1000:.3f} ms")
    print(f"  Average time per operation: {avg_time*1000/5:.3f} ms")
    print(f"  Operations per second: {1/avg_time*5:.0f}")

def main():
    """Main demonstration function"""
    print("=" * 60)
    
    try:
        # Initialize neural CPU
        neural_cpu = demonstrate_basic_operations()
        
        # Run demonstrations
        demonstrate_batch_processing(neural_cpu)
        demonstrate_system_status(neural_cpu)
        demonstrate_benchmarking(neural_cpu)
        demonstrate_optimization(neural_cpu)
        demonstrate_real_world_scenario(neural_cpu)
        
        # Final status
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        
        final_status = neural_cpu.get_system_status()
        print(f"Total operations executed: {final_status['total_operations']}")
        print("Neural CPU demonstration completed successfully!")
        
        # Cleanup
        neural_cpu.cleanup()
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
