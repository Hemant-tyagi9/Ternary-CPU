#!/usr/bin/env python3

import sys
import time
from main import TernaryCPU
from Cpu_components.ternary_gates import *

class TernaryCPUTestSuite:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def run_test(self, test_name, test_function):
        """Run a single test and track results"""
        print(f"\nRunning {test_name}...")
        print("-" * 50)
        
        try:
            start_time = time.perf_counter()
            result = test_function()
            end_time = time.perf_counter()
            
            if result:
                print(f"PASS: {test_name} ({end_time - start_time:.4f}s)")
                self.tests_passed += 1
                self.test_results.append((test_name, True, end_time - start_time))
            else:
                print(f"FAIL: {test_name}")
                self.tests_failed += 1
                self.test_results.append((test_name, False, end_time - start_time))
                
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            self.tests_failed += 1
            self.test_results.append((test_name, False, 0))
        
        return result
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        cpu = TernaryCPU()
        
        program = [
            ("LOADI", 1, 0, 0),    # R0 = 1
            ("LOADI", 2, 0, 1),    # R1 = 2
            ("ADD", 0, 1, 2),      # R2 = R0 + R1 = 0 (1+2=3, 3%3=0)
            ("SUB", 1, 0, 3),      # R3 = R1 - R0 = 1 (2-1=1)
            ("LOADI", 0, 0, 4),    # R4 = 0
            ("ADD", 3, 4, 5),      # R5 = R3 + R4 = 1 (1+0=1)
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False)
        
        expected_regs = [1, 2, 0, 1, 0, 1, 0, 0, 0]
        actual_regs = result['regs']
        
        print(f"Expected registers: {expected_regs}")
        print(f"Actual registers:   {actual_regs}")
        print(f"Cycles: {result['cycles']}, Time: {result['time_s']*1000:.3f}ms")
        
        return actual_regs == expected_regs and result['halted']
    
    def test_logic_operations(self):
        """Test ternary logic operations"""
        cpu = TernaryCPU()
        
        program = [
            ("LOADI", 1, 0, 0),    # R0 = 1
            ("LOADI", 2, 0, 1),    # R1 = 2
            ("AND", 0, 1, 2),      # R2 = min(1,2) = 1
            ("OR", 0, 1, 3),       # R3 = max(1,2) = 2  
            ("XOR", 0, 1, 4),      # R4 = (1-2)%3 = 2
            ("LOADI", 0, 0, 5),    # R5 = 0
            ("OR", 5, 0, 6),       # R6 = max(0,1) = 1
            ("AND", 5, 1, 7),      # R7 = min(0,2) = 0
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False)
        
        expected_regs = [1, 2, 1, 2, 2, 0, 1, 0, 0]
        actual_regs = result['regs']
        
        print(f"Expected registers: {expected_regs}")
        print(f"Actual registers:   {actual_regs}")
        print(f"Cycles: {result['cycles']}, Time: {result['time_s']*1000:.3f}ms")
        
        return actual_regs == expected_regs and result['halted']
    
    def test_data_movement(self):
        """Test data movement operations"""
        cpu = TernaryCPU()
        
        program = [
            ("LOADI", 2, 0, 0),    # R0 = 2
            ("MOV", 0, 1, 0),      # R1 = R0 = 2
            ("MOV", 1, 2, 0),      # R2 = R1 = 2
            ("LOADI", 1, 0, 3),    # R3 = 1
            ("MOV", 3, 4, 0),      # R4 = R3 = 1
            ("ADD", 2, 4, 5),      # R5 = R2 + R4 = 2 + 1 = 0
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False)
        
        expected_regs = [2, 2, 2, 1, 1, 0, 0, 0, 0]
        actual_regs = result['regs']
        
        print(f"Expected registers: {expected_regs}")
        print(f"Actual registers:   {actual_regs}")
        print(f"Cycles: {result['cycles']}, Time: {result['time_s']*1000:.3f}ms")
        
        return actual_regs == expected_regs and result['halted']
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        cpu = TernaryCPU()
        
        program = [
            ("LOADI", 3, 0, 0),    # R0 = 3%3 = 0 (overflow handling)
            ("LOADI", 4, 0, 1),    # R1 = 4%3 = 1 (overflow handling)
            ("LOADI", 5, 0, 2),    # R2 = 5%3 = 2 (overflow handling)
            ("ADD", 0, 1, 3),      # R3 = 0 + 1 = 1
            ("ADD", 1, 2, 4),      # R4 = 1 + 2 = 0 (3%3=0)
            ("SUB", 0, 2, 5),      # R5 = 0 - 2 = -2%3 = 1
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False)
        
        expected_regs = [0, 1, 2, 1, 0, 1, 0, 0, 0]
        actual_regs = result['regs']
        
        print(f"Expected registers: {expected_regs}")
        print(f"Actual registers:   {actual_regs}")
        print(f"Cycles: {result['cycles']}, Time: {result['time_s']*1000:.3f}ms")
        
        return actual_regs == expected_regs and result['halted']
    
    def test_complex_program(self):
        """Test a more complex program with multiple operations"""
        cpu = TernaryCPU()
        
        # Program that calculates factorial-like operation in ternary
        program = [
            ("LOADI", 2, 0, 0),    # R0 = 2 (initial value)
            ("LOADI", 1, 0, 1),    # R1 = 1 (decrement value)
            ("LOADI", 0, 0, 2),    # R2 = 0 (accumulator)
            ("LOADI", 0, 0, 3),    # R3 = 0 (counter)
            # Loop start (address 4)
            ("ADD", 2, 0, 2),      # R2 = R2 + R0 (accumulate)
            ("SUB", 0, 1, 0),      # R0 = R0 - R1 (decrement)
            ("ADD", 3, 1, 3),      # R3 = R3 + 1 (increment counter)
            ("OR", 0, 0, 4),       # R4 = R0 (copy for testing)
            ("SUB", 4, 4, 5),      # R5 = 0 (zero for comparison)
            ("XOR", 4, 5, 6),      # R6 = R4 XOR 0 (non-zero if R0 != 0)
            ("LOADI", 4, 0, 7),    # R7 = 4 (loop address)
            ("JEQ", 12, 0, 0),     # Jump to end if R6 == 0
            ("JMP", 4, 0, 0),      # Jump back to loop start
            ("HALT", 0, 0, 0)      # End
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False, max_cycles=100)
        
        print(f"Final registers: {result['regs']}")
        print(f"Cycles: {result['cycles']}, Time: {result['time_s']*1000:.3f}ms")
        print(f"Halted: {result['halted']}")
        
        # Check that program executed and halted properly
        return result['halted'] and result['cycles'] > 10
    
    def test_memory_bounds(self):
        """Test memory boundary conditions"""
        cpu = TernaryCPU(memory_size=10)  # Small memory for testing
        
        # Program that fits in memory
        small_program = [
            ("LOADI", 1, 0, 0),
            ("LOADI", 2, 0, 1),
            ("ADD", 0, 1, 2),
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(small_program)
        result = cpu.run(verbose=False)
        
        print(f"Small program - Cycles: {result['cycles']}, Halted: {result['halted']}")
        
        # Test with program larger than memory
        cpu2 = TernaryCPU(memory_size=3)
        large_program = [
            ("LOADI", 1, 0, 0),
            ("LOADI", 2, 0, 1),
            ("ADD", 0, 1, 2),
            ("SUB", 2, 0, 3),
            ("HALT", 0, 0, 0)
        ]
        
        cpu2.load_program(large_program)  # Should truncate
        result2 = cpu2.run(verbose=False)
        
        print(f"Large program (truncated) - Cycles: {result2['cycles']}, Halted: {result2['halted']}")
        
        return result['halted'] and result2['halted']
    
    def test_register_bounds(self):
        """Test register boundary conditions"""
        cpu = TernaryCPU()
        
        program = [
            ("LOADI", 1, 0, 8),    # R8 = 1 (last valid register)
            ("LOADI", 2, 0, 7),    # R7 = 2
            ("ADD", 8, 7, 6),      # R6 = R8 + R7 = 0
            ("MOV", 6, 5, 0),      # R5 = R6 = 0
            ("HALT", 0, 0, 0)
        ]
        
        cpu.load_program(program)
        result = cpu.run(verbose=False)
        
        expected_regs = [0, 0, 0, 0, 0, 0, 0, 2, 1]
        actual_regs = result['regs']
        
        print(f"Expected registers: {expected_regs}")
        print(f"Actual registers:   {actual_regs}")
        
        return actual_regs == expected_regs
    
    def test_performance_comparison(self):
        """Test performance with different execution modes"""
        cpu = TernaryCPU()
        
        # Create a moderately complex program
        program = []
        for i in range(10):
            program.extend([
                ("LOADI", i % 3, 0, 0),
                ("LOADI", (i + 1) % 3, 0, 1),
                ("ADD", 0, 1, 2),
                ("SUB", 2, 0, 3),
                ("AND", 1, 3, 4),
                ("OR", 4, 0, 5)
            ])
        program.append(("HALT", 0, 0, 0))
        
        # Test verbose mode
        cpu.load_program(program)
        start_time = time.perf_counter()
        result_verbose = cpu.run(verbose=True)
        verbose_time = time.perf_counter() - start_time
        
        # Reset and test fast mode
        cpu = TernaryCPU()
        cpu.load_program(program)
        start_time = time.perf_counter()
        result_fast = cpu.run(verbose=False)
        fast_time = time.perf_counter() - start_time
        
        print(f"Verbose mode: {result_verbose['cycles']} cycles in {verbose_time*1000:.3f}ms")
        print(f"Fast mode:    {result_fast['cycles']} cycles in {fast_time*1000:.3f}ms")
        print(f"Speedup: {verbose_time/fast_time:.1f}x")
        
        # Both should produce same results
        return (result_verbose['regs'] == result_fast['regs'] and 
                result_verbose['cycles'] == result_fast['cycles'] and
                fast_time < verbose_time)
    
    def test_ternary_gates_integration(self):
        """Test integration with ternary gates module"""
        print("Testing ternary gate operations...")
        
        # Test all basic gates
        test_cases = [
            (ternary_and, 1, 2, 1),
            (ternary_or, 1, 2, 2),
            (ternary_xor, 1, 2, 2),
            (ternary_not, 1, None, 1),
            (ternary_eq, 2, 2, 2),
            (ternary_eq, 1, 2, 0)
        ]
        
        all_passed = True
        for gate_func, a, b, expected in test_cases:
            if b is not None:
                result = gate_func(a, b)
                test_name = f"{gate_func.__name__}({a}, {b})"
            else:
                result = gate_func(a)
                test_name = f"{gate_func.__name__}({a})"
            
            if result == expected:
                print(f"  PASS: {test_name} = {result}")
            else:
                print(f"  FAIL: {test_name} = {result}, expected {expected}")
                all_passed = False
        
        # Test addition with carry
        result, carry = ternary_add(2, 2)
        print(f"  ternary_add(2, 2) = ({result}, {carry}) - {'PASS' if result == 1 and carry == 1 else 'FAIL'}")
        
        return all_passed
    
    def run_all_tests(self):
        """Run all tests in the suite"""
        print("=" * 60)
        print("TERNARY CPU COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        test_functions = [
            ("Basic Arithmetic Operations", self.test_basic_arithmetic),
            ("Logic Operations", self.test_logic_operations),
            ("Data Movement Operations", self.test_data_movement),
            ("Edge Cases and Boundary Conditions", self.test_edge_cases),
            ("Complex Program Execution", self.test_complex_program),
            ("Memory Boundary Conditions", self.test_memory_bounds),
            ("Register Boundary Conditions", self.test_register_bounds),
            ("Performance Comparison", self.test_performance_comparison),
            ("Ternary Gates Integration", self.test_ternary_gates_integration)
        ]
        
        for test_name, test_func in test_functions:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed:    {self.tests_passed}")
        print(f"Tests failed:    {self.tests_failed}")
        print(f"Pass rate:       {pass_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, passed, duration in self.test_results:
            status = "PASS" if passed else "FAIL"
            print(f"  {status:4} | {test_name:35} | {duration:.4f}s")
        
        if self.tests_failed == 0:
            print("\n*** ALL TESTS PASSED! ***")
            print("The ternary CPU implementation is working correctly.")
        else:
            print(f"\n*** {self.tests_failed} TEST(S) FAILED ***")
            print("Please review the failed tests and fix any issues.")
        
        return self.tests_failed == 0

def benchmark_ternary_vs_binary():
    """Compare ternary CPU performance against equivalent binary operations"""
    print("\n" + "=" * 60)
    print("TERNARY VS BINARY PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Ternary CPU test
    ternary_cpu = TernaryCPU()
    ternary_program = [
        ("LOADI", 1, 0, 0),
        ("LOADI", 2, 0, 1),
        ("ADD", 0, 1, 2),
        ("AND", 0, 1, 3),
        ("OR", 0, 1, 4),
        ("XOR", 0, 1, 5),
        ("HALT", 0, 0, 0)
    ]
    
    ternary_cpu.load_program(ternary_program)
    start_time = time.perf_counter()
    ternary_result = ternary_cpu.run(verbose=False)
    ternary_time = time.perf_counter() - start_time
    
    # Equivalent binary operations (simulated)
    start_time = time.perf_counter()
    binary_ops = []
    for _ in range(ternary_result['cycles']):
        a, b = 1, 0  # Binary equivalents
        result1 = a + b
        result2 = a & b  
        result3 = a | b
        result4 = a ^ b
        binary_ops.extend([result1, result2, result3, result4])
    binary_time = time.perf_counter() - start_time
    
    print(f"Ternary CPU:")
    print(f"  Cycles: {ternary_result['cycles']}")
    print(f"  Time: {ternary_time*1000:.3f}ms")
    print(f"  Final registers: {ternary_result['regs']}")
    
    print(f"\nBinary Operations (equivalent):")
    print(f"  Operations: {len(binary_ops)}")
    print(f"  Time: {binary_time*1000:.3f}ms")
    
    print(f"\nComparison:")
    if ternary_time > 0:
        print(f"  Speed ratio: {binary_time/ternary_time:.2f}x")
        print(f"  Ternary data density: 1.585 bits/symbol vs 1.0 for binary")

if __name__ == "__main__":
    # Run the comprehensive test suite
    test_suite = TernaryCPUTestSuite()
    success = test_suite.run_all_tests()
    
    # Run performance benchmark
    benchmark_ternary_vs_binary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
