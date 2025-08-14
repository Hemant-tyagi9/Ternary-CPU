import sys
import os
from typing import List, Tuple, Dict, Any
import time

class TernaryMemory:
    def __init__(self, size=243):  # 3^5 for better addressing
        self.memory = [("NOP", 0, 0, 0) for _ in range(size)]  # Initialize with NOP instructions
        self.size = size
        
    def load(self, address):
        if 0 <= address < self.size:
            return self.memory[address]
        return ("HALT", 0, 0, 0)  # Return HALT if out of bounds
        
    def store(self, address, value):
        if 0 <= address < self.size:
            self.memory[address] = value

class RegisterSet:
    def __init__(self, num_registers=9):
        self.registers = [0] * num_registers
    
    def read(self, index):
        if 0 <= index < len(self.registers):
            return self.registers[index] % 3
        return 0
    
    def write(self, index, value):
        if 0 <= index < len(self.registers):
            self.registers[index] = value % 3

class ProgramCounter:
    def __init__(self, max_value=242):
        self.value = 0
        self.max_value = max_value

    def increment(self):
        self.value = min(self.value + 1, self.max_value)

    def jump(self, address):
        if 0 <= address <= self.max_value:
            self.value = address
        else:
            raise ValueError(f"Address {address} out of range [0, {self.max_value}]")

    def get(self):
        return self.value

    def reset(self):
        self.value = 0

class TernaryALU:
    def __init__(self):
        self.flags = {
            "zero": False,
            "carry": False,
            "negative": False,
            "overflow": False
        }
        
    def _update_flags(self, result):
        result = result % 3
        self.flags["zero"] = (result == 0)
        self.flags["negative"] = (result == 2)
        return result
        
    def execute(self, op, a, b=None):
        a = a % 3
        if b is not None:
            b = b % 3
            
        if op == "ADD":
            result = (a + b) % 3
        elif op == "SUB":
            result = (a - b) % 3
        elif op == "AND":
            result = min(a, b)
        elif op == "OR":
            result = max(a, b)
        elif op == "XOR":
            result = (a - b) % 3 if a != b else 0
        elif op == "NOT":
            result = 2 - a
        else:
            result = 0
            
        return self._update_flags(result)

class TernaryCPU:
    def __init__(self, memory_size=243):
        self.memory = TernaryMemory(size=memory_size)
        self.registers = RegisterSet(num_registers=9)
        self.pc = ProgramCounter(max_value=memory_size-1)
        self.alu = TernaryALU()
        self.halted = False
        self.debug = False
        self.cycle_count = 0

    def load_program(self, program):
        """Load program into memory with bounds checking"""
        if self.debug:
            print("\nLoading program:")
        
        for addr, instr in enumerate(program):
            if addr >= self.memory.size:
                if self.debug:
                    print(f"Warning: Program truncated at instruction {addr}")
                break
            self.memory.store(addr, instr)
            if self.debug:
                print(f"  [{addr:2d}] {instr}")

    def step(self):
        """Execute one instruction with debug output"""
        if self.halted:
            if self.debug:
                print("CPU is halted")
            return False

        addr = self.pc.get()
        if addr >= self.memory.size:
            if self.debug:
                print(f"PC {addr} out of bounds, halting")
            self.halted = True
            return False

        instr = self.memory.load(addr)
        if instr is None:
            if self.debug:
                print(f"Invalid instruction at {addr}, halting")
            self.halted = True
            return False

        if self.debug:
            print(f"\n[Cycle {self.cycle_count+1}] PC={addr}: {instr}")

        op = instr[0]
        if op == "LOADI":
            self.cycle_count += 1
            reg = instr[3]
            value = instr[1] % 3
            self.registers.write(reg, value)
            if self.debug:
                print(f"  LOADI: R{reg} <- {value}")

        elif op == "ADD":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            res = self.alu.execute("ADD", a, b)
            self.registers.write(dest, res)
            if self.debug:
                print(f"  ADD: R{dest} <- R{src1}({a}) + R{src2}({b}) = {res}")

        elif op == "SUB":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            res = self.alu.execute("SUB", a, b)
            self.registers.write(dest, res)
            if self.debug:
                print(f"  SUB: R{dest} <- R{src1}({a}) - R{src2}({b}) = {res}")

        elif op == "AND":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            res = self.alu.execute("AND", a, b)
            self.registers.write(dest, res)
            if self.debug:
                print(f"  AND: R{dest} <- R{src1}({a}) AND R{src2}({b}) = {res}")

        elif op == "OR":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            res = self.alu.execute("OR", a, b)
            self.registers.write(dest, res)
            if self.debug:
                print(f"  OR: R{dest} <- R{src1}({a}) OR R{src2}({b}) = {res}")

        elif op == "XOR":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            res = self.alu.execute("XOR", a, b)
            self.registers.write(dest, res)
            if self.debug:
                print(f"  XOR: R{dest} <- R{src1}({a}) XOR R{src2}({b}) = {res}")

        elif op == "MOV":
            self.cycle_count += 1
            src = instr[1]
            dest = instr[2]
            value = self.registers.read(src)
            self.registers.write(dest, value)
            if self.debug:
                print(f"  MOV: R{dest} <- R{src}({value})")

        elif op == "JMP":
            self.cycle_count += 1
            target = instr[1]
            self.pc.jump(target)
            if self.debug:
                print(f"  JMP: PC <- {target}")
            return True  # Skip normal PC increment

        elif op == "JEQ":
            self.cycle_count += 1
            target = instr[1]
            if self.alu.flags["zero"]:
                self.pc.jump(target)
                if self.debug:
                    print(f"  JEQ: PC <- {target} (taken)")
                return True
            if self.debug:
                print(f"  JEQ: Not taken")

        elif op == "NOP":
            self.cycle_count += 1
            if self.debug:
                print(f"  NOP: No operation")

        elif op == "HALT":
            if self.debug:
                print("  HALT instruction encountered")
            self.halted = True
            return False

        else:
            if self.debug:
                print(f"  Unknown instruction: {op}")
            self.halted = True
            return False

        self.pc.increment()
        return True

    def run(self, max_cycles=10000, verbose=False):
        """Run the CPU with optimized execution"""
        self.debug = verbose
        start_time = time.perf_counter()
        self.cycle_count = 0
        self.halted = False
        self.pc.reset()

        if verbose:
            # Slow debug mode with step-by-step execution
            while not self.halted and self.cycle_count < max_cycles:
                if not self.step():
                    break
        else:
            # Fast optimized execution
            while not self.halted and self.cycle_count < max_cycles:
                addr = self.pc.get()
                if addr >= self.memory.size:
                    self.halted = True
                    break
                    
                instr = self.memory.load(addr)
                op = instr[0]

                if op == "LOADI":
                    self.registers.write(instr[3], instr[1] % 3)
                elif op == "ADD":
                    a = self.registers.read(instr[1])
                    b = self.registers.read(instr[2])
                    result = (a + b) % 3
                    self.registers.write(instr[3], result)
                elif op == "SUB":
                    a = self.registers.read(instr[1])
                    b = self.registers.read(instr[2])
                    result = (a - b) % 3
                    self.registers.write(instr[3], result)
                elif op == "AND":
                    a = self.registers.read(instr[1])
                    b = self.registers.read(instr[2])
                    result = min(a, b)
                    self.registers.write(instr[3], result)
                elif op == "OR":
                    a = self.registers.read(instr[1])
                    b = self.registers.read(instr[2])
                    result = max(a, b)
                    self.registers.write(instr[3], result)
                elif op == "XOR":
                    a = self.registers.read(instr[1])
                    b = self.registers.read(instr[2])
                    result = (a - b) % 3 if a != b else 0
                    self.registers.write(instr[3], result)
                elif op == "MOV":
                    value = self.registers.read(instr[1])
                    self.registers.write(instr[2], value)
                elif op == "JMP":
                    self.pc.jump(instr[1])
                    self.cycle_count += 1
                    continue
                elif op == "NOP":
                    pass
                elif op == "HALT":
                    self.halted = True
                    break
                else:
                    self.halted = True
                    break

                self.pc.increment()
                self.cycle_count += 1

        end_time = time.perf_counter()
        
        return {
            "halted": self.halted,
            "cycles": self.cycle_count,
            "time_s": end_time - start_time,
            "pc": self.pc.get(),
            "regs": [self.registers.read(i) for i in range(9)],
            "flags": self.alu.flags.copy()
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current CPU state for debugging"""
        return {
            "pc": self.pc.get(),
            "registers": [self.registers.read(i) for i in range(9)],
            "flags": self.alu.flags.copy(),
            "halted": self.halted,
            "cycles": self.cycle_count
        }

def create_test_programs():
    """Create various test programs for the ternary CPU"""
    
    # Basic arithmetic test
    basic_test = [
        ("LOADI", 1, 0, 0),    # R0 = 1
        ("LOADI", 2, 0, 1),    # R1 = 2  
        ("ADD", 0, 1, 2),      # R2 = R0 + R1 = 0 (1+2=3, 3%3=0)
        ("LOADI", 1, 0, 3),    # R3 = 1
        ("SUB", 1, 0, 4),      # R4 = R1 - R0 = 1 (2-1=1)
        ("HALT", 0, 0, 0)
    ]
    
    # Logic operations test
    logic_test = [
        ("LOADI", 1, 0, 0),    # R0 = 1
        ("LOADI", 2, 0, 1),    # R1 = 2
        ("AND", 0, 1, 2),      # R2 = min(1,2) = 1
        ("OR", 0, 1, 3),       # R3 = max(1,2) = 2
        ("XOR", 0, 1, 4),      # R4 = 1-2 = -1%3 = 2
        ("HALT", 0, 0, 0)
    ]
    
    # Loop test with jump
    loop_test = [
        ("LOADI", 0, 0, 0),    # R0 = 0 (counter)
        ("LOADI", 1, 0, 1),    # R1 = 1 (increment)
        ("LOADI", 2, 0, 2),    # R2 = 2 (limit)
        ("ADD", 0, 1, 0),      # R0 = R0 + R1 (increment counter)
        ("SUB", 2, 0, 3),      # R3 = R2 - R0 (check if done)
        ("JEQ", 7, 0, 0),      # Jump to HALT if R3 == 0
        ("JMP", 3, 0, 0),      # Jump back to increment
        ("HALT", 0, 0, 0)
    ]
    
    return {
        "basic": basic_test,
        "logic": logic_test,
        "loop": loop_test
    }

def run_comprehensive_tests():
    """Run comprehensive tests on the ternary CPU"""
    print("Ternary CPU Comprehensive Test Suite")
    print("=" * 50)
    
    test_programs = create_test_programs()
    
    for test_name, program in test_programs.items():
        print(f"\nRunning {test_name.upper()} test:")
        print("-" * 30)
        
        cpu = TernaryCPU()
        cpu.load_program(program)
        
        # Run with verbose output for first test
        verbose = (test_name == "basic")
        result = cpu.run(verbose=verbose)
        
        print(f"Test completed:")
        print(f"  Cycles: {result['cycles']}")
        print(f"  Time: {result['time_s']*1000:.3f}ms")
        print(f"  Final PC: {result['pc']}")
        print(f"  Registers: {result['regs']}")
        print(f"  Halted: {result['halted']}")
        
        if test_name == "basic":
            # Verify expected results
            expected_regs = [1, 2, 0, 1, 1, 0, 0, 0, 0]
            if result['regs'] == expected_regs:
                print("  PASSED: Register values match expected")
            else:
                print(f"  FAILED: Expected {expected_regs}, got {result['regs']}")

if __name__ == "__main__":
    run_comprehensive_tests()
