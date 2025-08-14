import time
from Integration.cpu_extend import TernaryCPU

class BinaryCPU:
    def __init__(self):
        self.reg = [0]*8
        self.pc = 0
        self.cycles = 0
        
    def load_program(self, program):
        self.program = program
        
    def step(self):
        if self.pc >= len(self.program):
            return False
            
        op, *args = self.program[self.pc]
        if op == "LOADI":
            self.reg[args[2]] = args[0] & 1
        elif op == "AND":
            self.reg[args[2]] = self.reg[args[0]] & self.reg[args[1]]
        elif op == "HALT":
            return False
            
        self.pc += 1
        self.cycles += 1
        return True
        
    def run(self):
        start = time.perf_counter()
        while self.step():
            pass
        return {
            "cycles": self.cycles,
            "time_s": time.perf_counter() - start,
            "regs": self.reg.copy()
        }

def compare_performance():
    # Same test program for both CPUs
    test_program = [
        ("LOADI", 1, 0, 0),
        ("LOADI", 1, 0, 1),
        ("AND", 0, 1, 2),
        ("HALT", 0, 0, 0)
    ]
    
    # Run binary CPU
    binary_cpu = BinaryCPU()
    binary_cpu.load_program(test_program)
    binary_result = binary_cpu.run()
    
    # Run ternary CPU
    ternary_cpu = TernaryCPU()
    ternary_cpu.load_program(test_program)
    ternary_result = ternary_cpu.run()
    
    print("Binary CPU:")
    print(f"  Cycles: {binary_result['cycles']}")
    print(f"  Time: {binary_result['time_s']*1000:.3f}ms")
    
    print("\nTernary CPU:")
    print(f"  Cycles: {ternary_result['cycles']}")
    print(f"  Time: {ternary_result['time_s']*1000:.3f}ms")
    
    print(f"\nTernary is {binary_result['time_s']/ternary_result['time_s']:.2f}x faster")

if __name__ == "__main__":
    compare_performance()
