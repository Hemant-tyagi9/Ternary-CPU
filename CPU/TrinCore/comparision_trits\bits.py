import numpy as np
import matplotlib.pyplot as plt
from Cpu_components.ternary_gates import ternary_and, ternary_or
from timeit import timeit
import psutil

def binary_and(a, b):
    return a & b

def binary_or(a, b):
    return a | b

def measure_energy(operation, inputs, ternary=True):
    """Measure energy consumption of operations"""
    start_energy = psutil.cpu_percent(interval=0.1)
    if ternary:
        [operation(a%3, b%3) for a, b in inputs]
    else:
        [operation(a%2, b%2) for a, b in inputs]
    end_energy = psutil.cpu_percent(interval=0.1)
    return end_energy - start_energy

def benchmark():
    # Generate test data
    inputs = [(a, b) for a in range(3) for b in range(3)]
    binary_inputs = [(a%2, b%2) for a in range(3) for b in range(3)]
    
    # Time benchmarks
    ternary_time = timeit(lambda: [ternary_and(a, b) for a, b in inputs], number=1000)
    binary_time = timeit(lambda: [binary_and(a, b) for a, b in binary_inputs], number=1000)
    
    # Energy benchmarks
    ternary_energy = measure_energy(ternary_and, inputs)
    binary_energy = measure_energy(binary_and, binary_inputs, ternary=False)
    
    # Results
    results = {
        'ternary_time': ternary_time,
        'binary_time': binary_time,
        'ternary_energy': ternary_energy,
        'binary_energy': binary_energy,
        'speed_ratio': binary_time / ternary_time,
        'energy_ratio': binary_energy / ternary_energy
    }
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.bar(['Ternary', 'Binary'], [ternary_time, binary_time])
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (s)')
    
    # Energy comparison
    ax2.bar(['Ternary', 'Binary'], [ternary_energy, binary_energy])
    ax2.set_title('Energy Consumption Comparison')
    ax2.set_ylabel('Energy (%)')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    results = benchmark()
    print("Performance Comparison Results:")
    print(f"Ternary operations were {results['speed_ratio']:.2f}x faster")
    print(f"Ternary operations used {results['energy_ratio']:.2f}x less energy")
