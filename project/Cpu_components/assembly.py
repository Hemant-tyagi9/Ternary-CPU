from enum import Enum

def assemble(program_lines):
    assembled = []
    for line in program_lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        opcode = parts[0].upper()
        args = [int(x) for x in parts[1:]]
        assembled.append((opcode, *args))
    return assembled

class OpCode(Enum):
    # Data Movement
    LOAD = 0    # LOAD Rx, addr    - Load from memory to register
    STORE = 1   # STORE Rx, addr   - Store register to memory
    MOV = 2     # MOV Rx, Ry       - Move register to register

    # Arithmetic
    ADD = 3     # ADD Rx, Ry       - Add registers
    SUB = 4     # SUB Rx, Ry       - Subtract registers

    # Logic
    AND = 5     # AND Rx, Ry       - Ternary AND
    OR = 6      # OR Rx, Ry        - Ternary OR
    XOR = 7     # XOR Rx, Ry       - Ternary XOR

    # Control Flow
    JMP = 8     # JMP addr         - Unconditional jump
    JEQ = 9     # JEQ addr         - Jump if equal (zero flag)
    JNE = 10    # JNE addr         - Jump if not equal

    # Immediate Operations
    LOADI = 11  # LOADI Rx, value  - Load immediate value
    ADDI = 12   # ADDI Rx, value   - Add immediate value

    # System
    HLT = 13    # HLT              - Halt execution
    NOP = 14    # NOP              - No operation
    PRINT = 15  # PRINT Rx         - Print register value

