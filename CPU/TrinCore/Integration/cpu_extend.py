import sys
import os
from typing import List, Tuple
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Cpu_components.ternary_memory import TernaryMemory
from Cpu_components.register_set import RegisterSet
from Cpu_components.program_counter import ProgramCounter
from Cpu_components.alu import TernaryALU
from Cpu_components.assembly import OpCode

class TernaryCPU:
    def __init__(self, memory_size=27):
        self.memory = TernaryMemory(size=memory_size)
        self.registers = RegisterSet(num_registers=9)
        self.pc = ProgramCounter()
        self.alu = TernaryALU()
        self.halted = False
        self.debug = True
        self.cycle_count = 0

    def load_program(self, program):
        print("\nLoading program:")
        for addr, instr in enumerate(program):
            if addr >= len(self.memory.memory):
                print(f"Warning: Program truncated at instruction {addr}")
                break
            self.memory.store(addr, instr)
            print(f"  [{addr:2d}] {instr}")

    def step(self):
        """Slow debug step-by-step execution"""
        if self.halted:
            print("CPU is halted")
            return False

        addr = self.pc.get()
        if addr >= len(self.memory.memory):
            print(f"PC {addr} out of bounds, halting")
            self.halted = True
            return False

        instr = self.memory.load(addr)
        if instr is None:
            print(f"Invalid instruction at {addr}, halting")
            self.halted = True
            return False

        print(f"\n[Cycle {self.cycle_count+1}] PC={addr}: {instr}")

        op = instr[0]
        if op == "LOADI":
            self.cycle_count += 1
            reg = instr[3]
            value = instr[1] % 3
            self.registers.write(reg, value)
            print(f"  LOADI: R{reg} <- {value}")

        elif op == "ADD":
            self.cycle_count += 1
            src1 = instr[1]
            src2 = instr[2]
            dest = instr[3]
            a = self.registers.read(src1) % 3
            b = self.registers.read(src2) % 3
            res = (a + b) % 3
            self.registers.write(dest, res)
            print(f"  ADD: R{dest} <- R{src1}({a}) + R{src2}({b}) = {res}")

        elif op == "HALT":
            print("  HALT instruction encountered")
            self.halted = True
            return False

        self.pc.increment()
        return True

    def run(self, max_cycles=10000, verbose=False):
        """
        Runs the CPU.
        If verbose=True -> uses slow debug path (step-by-step).
        If verbose=False -> uses optimized fast loop.
        """
        self.debug = verbose
        start = time.perf_counter()
        self.cycle_count = 0

        if verbose:
            # Original slow debug mode
            while not self.halted and self.cycle_count < max_cycles:
                if not self.step():
                    break
        else:
            # Optimized execution path
            pc_val = self.pc.value
            regs = self.registers.registers  # direct list
            mem = self.memory.memory
            halted = False
            cycles = 0

            while not halted and cycles < max_cycles:
                instr = mem[pc_val]
                op = instr[0]

                if op == "LOADI":
                    val = instr[1]
                    if val >= 3:
                        val -= 3
                    regs[instr[3]] = val

                elif op == "ADD":
                    a = regs[instr[1]]
                    b = regs[instr[2]]
                    s = a + b
                    if s >= 3:
                        s -= 3
                    regs[instr[3]] = s

                elif op == "HALT":
                    halted = True
                    break

                pc_val += 1
                cycles += 1

            self.pc.value = pc_val
            self.cycle_count = cycles
            self.halted = halted

        return {
            "halted": self.halted,
            "cycles": self.cycle_count,
            "time_s": time.perf_counter() - start,
            "pc": self.pc.get(),
            "regs": [self.registers.read(i) % 3 for i in range(9)]  # Ensure ternary values
        }

if __name__ == "__main__":
    cpu = TernaryCPU(memory_size=7)
    test_program = [
        ("LOADI", 1, 0, 0),
        ("LOADI", 2, 0, 1),
        ("ADD", 0, 1, 2),
        ("LOADI", 0, 0, 3),
        ("LOADI", 1, 0, 4),
        ("ADD", 3, 4, 5),
        ("HALT", 0, 0, 0)
    ]
    cpu.load_program(test_program)
    result = cpu.run(verbose=True)
    print("\nFinal registers:", result['regs'])

