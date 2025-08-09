from typing import List, Tuple
import numpy as np
from Cpu_components.ternary_memory import TernaryMemory
from Cpu_components.register_set import RegisterSet
from Cpu_components.program_counter import ProgramCounter
from Cpu_components.alu import TernaryALU
from NN.nn_integration import NeuralIntegration
from Cpu_components.assembly import OpCode

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
        self.spiking_network = None

    # ===== Instruction Encoding/Decoding =====
    def _encode_instruction(self, instr: Tuple) -> int:
        """Convert (opcode_str, op1, op2, op3) into an integer for memory."""
        opcode_str = instr[0]
        try:
            opcode_id = OpCode[opcode_str].value
        except KeyError:
            opcode_id = OpCode.NOP.value
        # Fill missing operands with 0
        ops = list(instr[1:]) + [0] * (3 - len(instr[1:]))
        op1, op2, op3 = (o % 3 for o in ops)
        # Mixed radix encoding: opcode_id in high bits
        return (opcode_id << 6) | (op1 << 4) | (op2 << 2) | op3

    def _decode_instruction(self, code: int) -> Tuple:
        """Convert integer from memory back to (opcode_str, op1, op2, op3)."""
        if isinstance(code, (list, tuple)):
            return tuple(code)
        try:
            opcode_id = (code >> 6) & 0xFF
            op1 = (code >> 4) & 0x3
            op2 = (code >> 2) & 0x3
            op3 = code & 0x3
        except Exception:
            return ("NOP",)
        # Reverse lookup in OpCode
        opcode_str = next((name for name, member in OpCode.__members__.items()
                           if member.value == opcode_id), "NOP")
        return (opcode_str, op1, op2, op3)

    # ===== Program Loading =====
    def load_program(self, program: List[Tuple]):
        """Load program into memory with encoding."""
        for addr, instruction in enumerate(program):
            code = self._encode_instruction(instruction)
            self.memory.store(addr, code)

    # ===== Execution =====
    def execute(self, instruction: Tuple) -> int:
        opcode = instruction[0]
        self.operation_stats[opcode] = self.operation_stats.get(opcode, 0) + 1
        use_neural = (self.neural_mode and opcode in ["AND", "OR", "XOR"] and
                      self.operation_stats.get(opcode, 0) > 5)
        if use_neural:
            return self._neural_execute(instruction)
        else:
            return self._traditional_execute(instruction)

    def _traditional_execute(self, instruction):
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
        elif opcode in ("HLT", "HALT"):
            pass
        else:
            pass
        self.pc.increment()
        return 0
    

    def _neural_execute(self, instruction):
        opcode = instruction[0]
        if opcode in ["AND", "OR", "XOR"]:
            dest, src1, src2 = instruction[1], instruction[2], instruction[3]
            a = self.registers.read(src1)
            b = self.registers.read(src2)
            result = self.neural_alu.execute_operation(opcode, a, b)
            self.registers.write(dest, result)
        self.pc.increment()
        return 0
    
    def traditionalExecute(self, operation: str, a: int, b: int) -> int:
        operation = operation.upper()
        
        # Supported operations
        supported_ops = ["ADD", "SUB", "AND", "OR", "XOR", "NAND", "NOR"]
        if operation not in supported_ops:
            raise ValueError(f"Unsupported operation for traditionalExecute: {operation}")
        
        # Execute using ALU
        result = self.alu.execute(operation, a, b)
        
        # Ensure result is in ternary range
        return int(result % 3)

    # ===== Main Run Loop =====
    def run(self, max_cycles: int = 1000000, verbose: bool = False):
        cycles = 0
        halted = False
        while cycles < max_cycles:
            addr = self.pc.get()
            try:
                raw = self.memory.load(addr)
            except Exception:
                break
            instruction = self._decode_instruction(raw)
            opcode = instruction[0]
            if verbose:
                print(f"[CPU] PC={addr} OPCODE={opcode} INSTR={instruction}")
            if opcode in ("HLT", "HALT"):
                halted = True
                break
            try:
                self.execute(instruction)
            except Exception as e:
                if verbose:
                    print(f"[CPU] Exception: {e}")
                break
            cycles += 1
        if verbose:
            print(f"[CPU] run() finished - halted={halted}, cycles={cycles}, pc={self.pc.get()}")
        return {"halted": halted, "cycles": cycles, "pc": self.pc.get()}

