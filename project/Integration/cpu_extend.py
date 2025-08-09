from typing import List, Tuple, Dict, Any
import numpy as np
from Cpu_components.ternary_memory import TernaryMemory
from Cpu_components.register_set import RegisterSet
from Cpu_components.program_counter import ProgramCounter
from Cpu_components.alu import TernaryALU
from NN.nn_integration import NeuralIntegration

class TernaryCPU:
    """Neuromorphic ternary CPU with hybrid execution"""
    
    def __init__(self, memory_size=27, neural_mode=True):
        self.memory = TernaryMemory(size=memory_size)
        self.registers = RegisterSet(num_registers=9)
        self.pc = ProgramCounter()
        self.alu = TernaryALU()
        self.neural_mode = neural_mode
        self.neural_alu = NeuralIntegration() if neural_mode else None
        self.operation_stats = {}
        self.spiking_network = None  # Will hold neuromorphic component
        
    def load_program(self, program: List[Tuple]):
        """Load assembly program into memory"""
        for addr, instruction in enumerate(program):
            packed = self._pack_instruction(instruction)
            self.memory.store(addr, packed)
            
    def _pack_instruction(self, instruction):
        """Convert instruction to packed ternary format"""
        opcode_map = {
            "LOADI": 0,
            "ADD": 3,
            # Add all your opcodes here
            "HLT": 13
        }
        
        opcode = instruction[0]
        opcode_num = opcode_map.get(opcode, 14)  # Default to NOP
        
        if opcode == "LOADI":
            return (opcode_num << 4) | (instruction[1] << 2) | instruction[2]
        elif opcode == "ADD":
            return (opcode_num << 4) | (instruction[1] << 2) | instruction[2]
        # Add packing for other instruction formats
        else:
            return opcode_num << 4  # Instructions without operands
        
    def _unpack_instruction(self, packed_instruction):
        """Convert packed ternary instruction back to tuple format"""
        opcode_num = (packed_instruction >> 4) & 0xF  # First 4 bits are opcode
        operand1 = (packed_instruction >> 2) & 0x3   # Next 2 bits
        operand2 = packed_instruction & 0x3          # Last 2 bits
        
        # Map numeric opcodes back to strings
        opcode_map = {
            0: "LOADI",
            3: "ADD",
            # Add mappings for all your opcodes
        }
        
        opcode = opcode_map.get(opcode_num, "NOP")
        
        if opcode == "LOADI":
            return (opcode, operand1, operand2)
        elif opcode == "ADD":
            return (opcode, operand1, operand2, 0)  # Assuming dest is operand1
        else:
            return (opcode,)
            
    def execute(self, instruction: Tuple) -> int:
        """Execute a single instruction"""
        opcode = instruction[0]
        
        # Track operation statistics for adaptive execution
        self.operation_stats[opcode] = self.operation_stats.get(opcode, 0) + 1
        
        # Choose execution path based on mode and operation frequency
        use_neural = (self.neural_mode and 
                     opcode in ["AND", "OR", "XOR"] and 
                     self.operation_stats.get(opcode, 0) > 5)
        
        if use_neural:
            return self._neural_execute(instruction)
        else:
            return self._traditional_execute(instruction)
    
    def _traditional_execute(self, instruction):
        """Traditional ternary logic execution"""
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
        # Other operations...
        
        self.pc.increment()
        return 0
    
    def _neural_execute(self, instruction):
        """Neural network accelerated execution"""
        opcode = instruction[0]
        
        if opcode in ["AND", "OR", "XOR"]:
            src1, src2, dest = instruction[1], instruction[2], instruction[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            result = self.neural_alu.execute_operation(opcode, a, b)
            self.registers.write(dest, result)
        
        self.pc.increment()
        return 0
    
    def run(self):
        """Run the loaded program"""
        while True:
            addr = self.pc.get()
            instruction = self._unpack_instruction(self.memory.load(addr))
            if instruction[0] == "HLT":
                break
            self.execute(instruction)
