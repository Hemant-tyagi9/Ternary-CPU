from typing import List, Tuple, Any
from CPU_Components.ternary_memory import TernaryMemory
from CPU_Components.register_sets import RegisterSet
from CPU_Components.program_counter import ProgramCounter

class TernaryCPU:
    """Complete ternary CPU integration"""
    
    def __init__(self, memory: TernaryMemory, alu: Any, registers: RegisterSet, pc: ProgramCounter):
        self.memory = memory
        self.alu = alu
        self.registers = registers
        self.pc = pc
        self.running = False
        self.current_program = []
        
    def load_program(self, program: List[Tuple]):
        """Load a program into memory"""
        self.current_program = program
        for addr, instruction in enumerate(program):
            # Pack instruction into memory (simplified)
            packed = self._pack_instruction(instruction)
            self.memory.store(addr, packed)
    
    def _pack_instruction(self, instruction):
        """Convert instruction to numeric representation"""
        opcode = instruction[0]
        if opcode == "LOADI":
            return (0 << 4) | (instruction[1] << 2) | instruction[2]
        elif opcode == "ADD":
            return (3 << 4) | (instruction[1] << 2) | instruction[2]
        # Simplified packing for demo
        return 0
    
    def fetch(self):
        """Fetch next instruction"""
        addr = self.pc.get()
        return self.current_program[addr]
    
    def decode_execute(self, instruction):
        """Decode and execute instruction"""
        opcode = instruction[0]
        
        if opcode == "LOADI":
            reg, value = instruction[1], instruction[2]
            self.registers.write(reg, value)
        elif opcode == "ADD":
            dest, src1, src2 = instruction[1], instruction[2], instruction[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            result = self.alu.execute("ADD", a, b)
            self.registers.write(dest, result)
        elif opcode == "PRINT":
            reg = instruction[1]
            value = self.registers.read(reg)
            print(f"R{reg} = {value}")
        elif opcode == "HLT":
            self.running = False
        
        self.pc.increment()
    
    def run(self):
        """Run the loaded program"""
        self.running = True
        self.pc.reset()
        
        while self.running:
            instruction = self.fetch()
            self.decode_execute(instruction)
