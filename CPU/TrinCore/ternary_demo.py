#!/usr/bin/env python3

import sys
import time
from main import TernaryCPU
from Cpu_components.ternary_gates import *

def demo_basic_operations():
    """Demonstrate basic ternary CPU operations"""
    print("=" * 60)
    print("BASIC TERNARY CPU OPERATIONS DEMO")
    print("=" * 60)
    
    cpu = TernaryCPU()
    
    # Simple arithmetic demonstration
    print("\n1. Basic Arithmetic Operations:")
    print("-" * 30)
    
    program = [
        ("LOADI", 1, 0, 0),    # Load 1 into R0
        ("LOADI", 2, 0, 1),    # Load 2 into R1
        ("ADD", 0, 1, 2),      # R2 = R0 + R1 (1+2=3, 3%3=0 in ternary)
        ("SUB", 1, 0, 3),      # R3 = R1 - R0 (2-1=1)
        ("HALT", 0, 0, 0)
    ]
    
    cpu.load_program(program)
    result = cpu.run(verbose=True)
    
    print(f"\nResult: {result['regs'][:4]}")
    print("Note: In ternary arithmetic, 1+2=3, and 3 mod 3 = 0")

def demo_ternary_logic():
    """Demonstrate ternary logic operations"""
    print("\n=" * 60)
    print("TERNARY LOGIC OPERATIONS DEMO")
    print("=" * 60)
    
    cpu = TernaryCPU()
    
    print("\n2. Logic Operations with all ternary states:")
    print("-" * 45)
    
    program = [
        # Test with values 0, 1, 2
        ("LOADI", 0, 0, 0),    # R0 = 0
        ("LOADI", 1, 0, 1),    # R1 = 1  
        ("LOADI", 2, 0, 2),    # R2 = 2
        
        # AND operations: min(a,b)
        ("AND", 0, 1, 3),      # R3 = min(0,1) = 0
        ("AND", 1, 2, 4),      # R4 = min(1,2) = 1
        ("AND", 0, 2, 5),      # R5 = min(0,2) = 0
        
        # OR operations: max(a,b)  
        ("OR", 0, 1, 6),       # R6 = max(0,1) = 1
        ("OR", 1, 2, 7),       # R7 = max(1,2) = 2
        ("OR", 0, 2, 8),       # R8 = max(0,2) = 2
        
        ("HALT", 0, 0, 0)
    ]
    
    cpu.load_program(program)
    result = cpu.run(verbose=False)
    
    print(f"Input values:  R0={result['regs'][0]}, R1={result['regs'][1]}, R2={result['regs'][2]}")
    print(f"AND results:   min(0,1)={result['regs'][3]}, min(1,2)={result['regs'][4]}, min(0,2)={result['regs'][5]}")
    print(f"OR results:    max(0,1)={result['regs'][6]}, max(1,2)={result['regs'][7]}, max(0,2)={result['regs'][8]}")

def demo_data_density():
    """Demonstrate ternary data density advantages"""
    print("\n=" * 60)
    print("TERNARY DATA DENSITY DEMONSTRATION")
    print("=" * 60)
    
    print("\n3. Data Density Comparison:")
    print("-" * 30)
    
    # Binary system can represent 2^n states with n bits
    # Ternary system can represent 3^n states with n trits
    
    print("States representable with different symbol counts:")
    print("Symbols | Binary (2^n) | Ternary (3^n) | Ternary Advantage")
    print("--------|--------------|---------------|------------------")
    
    for n in range(1, 8):
        binary_states = 2**n
        ternary_states = 3**n
        advantage = ternary_states / binary_states
        print(f"   {n:2d}   |     {binary_states:4d}     |      {ternary_states:4d}      |     {advantage:.2f}x")
    
    print(f"\nInformation density per symbol:")
    print(f"Binary:  1.000 bits per bit")
    print(f"Ternary: {1.584962500721156:.3f} bits per trit (log₂(3))")
    print(f"Density advantage: {1.584962500721156:.1f}x")

def demo_complex_program():
    """Demonstrate a more complex ternary program"""
    print("\n=" * 60)
    print("COMPLEX TERNARY PROGRAM DEMONSTRATION")
    print("=" * 60)
    
    cpu = TernaryCPU()
    
    print("\n4. Ternary Number Processing:")
    print("-" * 30)
    print("Processing sequence: 0, 1, 2, 0, 1, 2...")
    
    # Program that processes a sequence of ternary numbers
    program = [
        ("LOADI", 0, 0, 0),    # R0 = 0 (current value)
        ("LOADI", 1, 0, 1),    # R1 = 1 (increment)
        ("LOADI", 0, 0, 2),    # R2 = 0 (sum accumulator)
        ("LOADI", 0, 0, 3),    # R3 = 0 (count)
        ("LOADI", 6, 0, 4),    # R4 = 6 (loop limit)
        
        # Loop body (addresses 5-11)
        ("ADD", 2, 0, 2),      # R2 += R0 (accumulate sum)
        ("ADD", 0, 1, 0),      # R0 = (R0 + 1) % 3 (next ternary digit)
        ("ADD", 3, 1, 3),      # R3++ (increment count)
        ("SUB", 4, 3, 5),      # R5 = limit - count
        ("XOR", 5, 5, 6),      # R6 = 0 if R5 == 0 (done?)
        ("JEQ", 12, 0, 0),     # Jump to end if done
        ("JMP", 5, 0, 0),      # Loop back
        
        ("HALT", 0, 0, 0)      # End
    ]
    
    cpu.load_program(program)
    result = cpu.run(verbose=False, max_cycles=50)
    
    print(f"Processed {result['regs'][3]} ternary digits")
    print(f"Final sum: {result['regs'][2]}")
    print(f"Last digit: {result['regs'][0]}")
    print(f"Executed in {result['cycles']} cycles")

def demo_ternary_gates():
    """Demonstrate standalone ternary gate operations"""
    print("\n=" * 60) 
    print("TERNARY LOGIC GATES DEMONSTRATION")
    print("=" * 60)
    
    print("\n5. Truth Tables for Ternary Gates:")
    print("-" * 35)
    
    # Basic gates
    gates = [
        ("AND", ternary_and),
        ("OR", ternary_or),
        ("XOR", ternary_xor),
        ("NAND", ternary_nand),
        ("NOR", ternary_nor)
    ]
    
    for gate_name, gate_func in gates:
        print(f"\n{gate_name} Gate:")
        print("A B | Result")
        print("----+-------")
        for a in range(3):
            for b in range(3):
                result = gate_func(a, b)
                print(f"{a} {b} |   {result}")
    
    # Unary gate
    print(f"\nNOT Gate:")
    print("A | Result")
    print("--+-------")
    for a in range(3):
        result = ternary_not(a)
        print(f"{a} |   {result}")

def performance_analysis():
    """Analyze performance characteristics"""
    print("\n=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("\n6. Execution Speed Comparison:")
    print("-" * 30)
    
    cpu = TernaryCPU()
    
    # Create test programs of different sizes
    test_sizes = [10, 50, 100, 200]
    
    for size in test_sizes:
        # Generate program with 'size' operations
        program = []
        for i in range(size):
            op_type = i % 5
            if op_type == 0:
                program.append(("LOADI", i % 3, 0, i % 9))
            elif op_type == 1:
                program.append(("ADD", (i % 8), (i+1) % 8, (i+2) % 9))
            elif op_type == 2:
                program.append(("SUB", (i % 8), (i+1) % 8, (i+2) % 9))
            elif op_type == 3:
                program.append(("AND", (i % 8), (i+1) % 8, (i+2) % 9))
            else:
                program.append(("OR", (i % 8), (i+1) % 8, (i+2) % 9))
        
        program.append(("HALT", 0, 0, 0))
        
        # Measure execution time
        cpu = TernaryCPU()
        cpu.load_program(program)
        
        start_time = time.perf_counter()
        result = cpu.run(verbose=False)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        ops_per_sec = result['cycles'] / execution_time if execution_time > 0 else 0
        
        print(f"Program size: {size:3d} ops | Cycles: {result['cycles']:3d} | "
              f"Time: {execution_time*1000:6.3f}ms | Speed: {ops_per_sec:.0f} ops/sec")

def interactive_demo():
    """Interactive demonstration allowing user input"""
    print("\n=" * 60)
    print("INTERACTIVE TERNARY CALCULATOR")
    print("=" * 60)
    
    print("\n7. Try your own ternary operations!")
    print("Enter two ternary digits (0, 1, or 2) and see the results.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("Enter operation (e.g., '1 + 2' or 'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            if '+' in user_input:
                parts = user_input.split('+')
                a, b = int(parts[0].strip()), int(parts[1].strip())
                result = (a + b) % 3
                print(f"  {a} + {b} = {result} (in ternary arithmetic)")
                
            elif '-' in user_input:
                parts = user_input.split('-')
                a, b = int(parts[0].strip()), int(parts[1].strip()) 
                result = (a - b) % 3
                print(f"  {a} - {b} = {result} (in ternary arithmetic)")
                
            elif 'and' in user_input.lower():
                parts = user_input.lower().split('and')
                a, b = int(parts[0].strip()), int(parts[1].strip())
                result = ternary_and(a, b)
                print(f"  {a} AND {b} = {result} (min function)")
                
            elif 'or' in user_input.lower():
                parts = user_input.lower().split('or')
                a, b = int(parts[0].strip()), int(parts[1].strip())
                result = ternary_or(a, b)
                print(f"  {a} OR {b} = {result} (max function)")
                
            else:
                print("  Format: 'a + b', 'a - b', 'a and b', or 'a or b'")
                
        except (ValueError, IndexError):
            print("  Please enter valid ternary digits (0, 1, or 2)")
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")

def main():
    """Main demonstration function"""
    print("TERNARY CPU COMPLETE DEMONSTRATION")
    print("Python Implementation of a 3-State Logic Processor")
    print("=" * 60)
    
    demos = [
        ("Basic Operations", demo_basic_operations),
        ("Ternary Logic", demo_ternary_logic),
        ("Data Density", demo_data_density),
        ("Complex Program", demo_complex_program),
        ("Logic Gates", demo_ternary_gates),
        ("Performance Analysis", performance_analysis),
        ("Interactive Calculator", interactive_demo)
    ]
    
    if len(sys.argv) > 1:
        # Run specific demo by name
        demo_name = sys.argv[1].lower()
        for name, func in demos:
            if demo_name in name.lower():
                func()
                return
        print(f"Demo '{demo_name}' not found.")
        print("Available demos:", [name for name, _ in demos])
    else:
        # Run all demos
        for name, func in demos:
            try:
                func()
                time.sleep(1)  # Pause between demos
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"Error in {name} demo: {e}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe ternary CPU implementation supports:")
        print("- Full ternary arithmetic (0, 1, 2 states)")
        print("- Ternary logic operations (AND, OR, XOR, etc.)")
        print("- 1.58x higher data density than binary")
        print("- Comprehensive instruction set")
        print("- Both verbose debugging and optimized execution")
        print("- Modular, extensible architecture")

if __name__ == "__main__":
    main()
