import time
import numpy as np
from typing import Dict, Any
from .nn import TernaryGateNN

def benchmark_neural_operation(model: TernaryGateNN, operation: str) -> Dict[str, Any]:
    """Benchmark a neural network implementation of a ternary operation"""
    test_cases = [(a, b) for a in range(3) for b in range(3)]
    times = []
    correct = 0
    
    for a, b in test_cases:
        # Get expected result
        if operation == "ADD":
            expected = (a + b) % 3
        elif operation == "SUB":
            expected = (a - b) % 3
        elif operation == "AND":
            expected = ternary_and(a, b)
        elif operation == "OR":
            expected = ternary_or(a, b)
        elif operation == "XOR":
            expected = ternary_xor(a, b)
        else:
            expected = 0
        
        # Time neural execution
        start_time = time.perf_counter()
        input_data = np.array([[a, b]])
        prediction = model.predict(input_data)[0]
        times.append(time.perf_counter() - start_time)
        
        # Check accuracy
        if prediction == expected:
            correct += 1
    
    return {
        'operation': operation,
        'accuracy': correct / len(test_cases),
        'avg_time': np.mean(times),
        'total_tests': len(test_cases)
    }

def benchmark_traditional_operation(operation: str) -> Dict[str, Any]:
    """Benchmark traditional implementation of a ternary operation"""
    test_cases = [(a, b) for a in range(3) for b in range(3)]
    times = []
    
    for a, b in test_cases:
        start_time = time.perf_counter()
        
        if operation == "ADD":
            result = (a + b) % 3
        elif operation == "SUB":
            result = (a - b) % 3
        elif operation == "AND":
            result = ternary_and(a, b)
        elif operation == "OR":
            result = ternary_or(a, b)
        elif operation == "XOR":
            result = ternary_xor(a, b)
        else:
            result = 0
        
        times.append(time.perf_counter() - start_time)
    
    return {
        'operation': operation,
        'avg_time': np.mean(times),
        'total_tests': len(test_cases)
    }
