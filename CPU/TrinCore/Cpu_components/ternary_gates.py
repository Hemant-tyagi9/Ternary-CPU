def ternary_and(a, b):
    """Ternary AND: returns minimum of inputs"""
    return min(a % 3, b % 3)

def ternary_or(a, b):
    """Ternary OR: returns maximum of inputs"""
    return max(a % 3, b % 3)

def ternary_xor(a, b):
    """Ternary XOR: exclusive or operation"""
    a, b = a % 3, b % 3
    return (a - b) % 3 if a != b else 0

def ternary_nand(a, b):
    """Ternary NAND: negated AND"""
    return 2 - ternary_and(a, b)

def ternary_nor(a, b):
    """Ternary NOR: negated OR"""
    return 2 - ternary_or(a, b)

def ternary_not(a):
    """Ternary NOT: negation"""
    return 2 - (a % 3)

def ternary_imply(a, b):
    """Ternary implication: a -> b"""
    a, b = a % 3, b % 3
    return max(2 - a, b)

def ternary_eq(a, b):
    """Ternary equality: returns 2 if equal, 0 if not"""
    return 2 if (a % 3) == (b % 3) else 0

def ternary_neq(a, b):
    """Ternary not equal: returns 0 if equal, 2 if not"""
    return 0 if (a % 3) == (b % 3) else 2

def ternary_majority(a, b, c):
    """Ternary majority gate: returns most common value"""
    a, b, c = a % 3, b % 3, c % 3
    values = [a, b, c]
    
    # Count occurrences of each value
    counts = {0: 0, 1: 0, 2: 0}
    for v in values:
        counts[v] += 1
    
    # Return the value with maximum count
    return max(counts, key=counts.get)

def ternary_mux(select, a, b):
    """Ternary multiplexer: selects input based on select signal"""
    select, a, b = select % 3, a % 3, b % 3
    if select == 0:
        return a
    elif select == 1:
        return b
    else:
        return min(a, b)  # Default case for select == 2

def ternary_add(a, b, carry_in=0):
    """Ternary addition with carry"""
    a, b, carry_in = a % 3, b % 3, carry_in % 3
    sum_val = a + b + carry_in
    carry_out = sum_val // 3
    result = sum_val % 3
    return result, carry_out

def ternary_sub(a, b, borrow_in=0):
    """Ternary subtraction with borrow"""
    a, b, borrow_in = a % 3, b % 3, borrow_in % 3
    diff = a - b - borrow_in
    if diff < 0:
        diff += 3
        borrow_out = 1
    else:
        borrow_out = 0
    return diff % 3, borrow_out

def ternary_inc(a):
    """Ternary increment"""
    return (a + 1) % 3

def ternary_dec(a):
    """Ternary decrement"""
    return (a - 1) % 3

def ternary_shift_left(a):
    """Ternary left shift (multiply by 3)"""
    # In ternary, left shift by 1 position multiplies by 3
    # For single trit, this wraps around
    return (a * 3) % 3

def ternary_shift_right(a):
    """Ternary right shift (divide by 3)"""
    # For single trit operations, this is essentially floor division
    return 0 if a == 0 else (a - 1) % 3

def ternary_rotate_left(a):
    """Ternary rotate left"""
    return (a * 3) % 3

def ternary_rotate_right(a):
    """Ternary rotate right"""
    return (a + 2) % 3

def balanced_ternary_add(a, b):
    """Balanced ternary addition (-1, 0, +1)"""
    # Convert from standard ternary (0,1,2) to balanced (-1,0,1)
    def to_balanced(x):
        return x - 1 if x == 2 else x
    
    def from_balanced(x):
        return x + 1 if x == -1 else x
    
    bal_a = to_balanced(a % 3)
    bal_b = to_balanced(b % 3)
    
    sum_val = bal_a + bal_b
    
    # Handle carries in balanced ternary
    if sum_val > 1:
        result = sum_val - 3
        carry = 1
    elif sum_val < -1:
        result = sum_val + 3
        carry = -1
    else:
        result = sum_val
        carry = 0
    
    return from_balanced(result), carry

def create_truth_table(operation, num_inputs=2):
    """Generate truth table for a ternary operation"""
    if num_inputs == 1:
        inputs = [(a,) for a in range(3)]
        results = [operation(a) for a in range(3)]
    elif num_inputs == 2:
        inputs = [(a, b) for a in range(3) for b in range(3)]
        results = [operation(a, b) for a in range(3) for b in range(3)]
    elif num_inputs == 3:
        inputs = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        results = [operation(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    else:
        raise ValueError("Unsupported number of inputs")
    
    return list(zip(inputs, results))

def benchmark_operations(iterations=10000):
    """Benchmark all ternary operations"""
    import time
    
    operations = {
        'AND': ternary_and,
        'OR': ternary_or,
        'XOR': ternary_xor,
        'NAND': ternary_nand,
        'NOR': ternary_nor,
        'NOT': ternary_not,
        'IMPLY': ternary_imply,
        'EQ': ternary_eq,
        'MAJORITY': ternary_majority,
        'MUX': ternary_mux
    }
    
    results = {}
    test_inputs_2 = [(a, b) for a in range(3) for b in range(3)]
    test_inputs_3 = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    
    for name, op in operations.items():
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            if name == 'NOT':
                for a in range(3):
                    op(a)
            elif name in ['MAJORITY', 'MUX']:
                for inputs in test_inputs_3:
                    op(*inputs)
            else:
                for inputs in test_inputs_2:
                    op(*inputs)
        
        end_time = time.perf_counter()
        results[name] = end_time - start_time
    
    return results

def validate_operations():
    """Validate all ternary operations with known results"""
    tests_passed = 0
    total_tests = 0
    
    # Test basic operations
    test_cases = [
        # AND tests
        (ternary_and, (0, 0), 0),
        (ternary_and, (1, 2), 1),
        (ternary_and, (2, 1), 1),
        
        # OR tests  
        (ternary_or, (0, 0), 0),
        (ternary_or, (1, 2), 2),
        (ternary_or, (0, 1), 1),
        
        # XOR tests
        (ternary_xor, (0, 0), 0),
        (ternary_xor, (1, 2), 2),
        (ternary_xor, (2, 1), 1),
        
        # NOT tests
        (ternary_not, (0,), 2),
        (ternary_not, (1,), 1),
        (ternary_not, (2,), 0),
        
        # EQ tests
        (ternary_eq, (1, 1), 2),
        (ternary_eq, (0, 1), 0),
        (ternary_eq, (2, 2), 2),
        
        # Addition tests
        (lambda a, b: ternary_add(a, b)[0], (1, 2), 0),  # 1+2=3, 3%3=0
        (lambda a, b: ternary_add(a, b)[0], (2, 2), 1),  # 2+2=4, 4%3=1
        (lambda a, b: ternary_add(a, b)[1], (2, 2), 1),  # carry = 4//3 = 1
    ]
    
    print("Validating ternary operations...")
    for operation, inputs, expected in test_cases:
        total_tests += 1
        try:
            result = operation(*inputs)
            if result == expected:
                tests_passed += 1
                print(f"PASS: {operation.__name__}{inputs} = {result}")
            else:
                print(f"FAIL: {operation.__name__}{inputs} = {result}, expected {expected}")
        except Exception as e:
            print(f"ERROR: {operation.__name__}{inputs} raised {e}")
    
    print(f"\nValidation complete: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

if __name__ == "__main__":
    print("Ternary Logic Gates Test Suite")
    print("=" * 40)
    
    # Validate operations
    validation_passed = validate_operations()
    
    # Show truth tables for basic operations
    print("\nTruth Tables:")
    print("-" * 20)
    
    basic_ops = [
        ("AND", ternary_and),
        ("OR", ternary_or), 
        ("XOR", ternary_xor),
        ("NOT", ternary_not)
    ]
    
    for name, op in basic_ops:
        print(f"\n{name}:")
        if name == "NOT":
            table = create_truth_table(op, 1)
            for inputs, result in table:
                print(f"  {name}({inputs[0]}) = {result}")
        else:
            table = create_truth_table(op, 2)
            for inputs, result in table:
                print(f"  {name}({inputs[0]}, {inputs[1]}) = {result}")
    
    # Benchmark operations
    print("\nBenchmarking operations...")
    benchmark_results = benchmark_operations(1000)
    
    print("\nBenchmark Results (1000 iterations):")
    for name, time_taken in sorted(benchmark_results.items(), key=lambda x: x[1]):
        print(f"  {name:10}: {time_taken:.6f}s")
    
    print(f"\nAll tests passed: {validation_passed}")
