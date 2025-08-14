"""
CPU Operation Benchmark
Measures performance of ternary vs binary operations
"""

import timeit
import numpy as np
from Cpu_components.ternary_gates import *

def binary_and(a, b):
    return a & b

def benchmark_operations():
    # Test cases
    ternary_inputs = [(a, b) for a in range(3) for b in range(3)]
    binary_inputs = [(a%2, b%2) for a in range(3) for b in range(3)]
    
    # Benchmark ternary
    ternary_time = timeit.timeit(
        lambda: [ternary_and(a, b) for a, b in ternary_inputs],
        number=10000
    )
    
    # Benchmark binary
    binary_time = timeit.timeit(
        lambda: [binary_and(a, b) for a, b in binary_inputs],
        number=10000
    )
    
    print(f"Ternary AND: {ternary_time:.4f}s (10k runs)")
    print(f"Binary AND: {binary_time:.4f}s (10k runs)")
    print(f"Speed ratio: {binary_time/ternary_time:.2f}x")

if __name__ == "__main__":
    benchmark_operations()
