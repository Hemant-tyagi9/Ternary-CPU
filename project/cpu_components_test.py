import unittest
from Cpu_components.alu import TernaryALU
from Cpu_components.register_set import RegisterSet
from Cpu_components.ternary_memory import TernaryMemory
from Cpu_components.program_counter import ProgramCounter
from Cpu_components.assembly import OpCode, assemble
from Integration.cpu_extend import TernaryCPU

class TestTernaryComponents(unittest.TestCase):
    def setUp(self):
        self.alu = TernaryALU()
        self.registers = RegisterSet()
        self.memory = TernaryMemory()
        self.pc = ProgramCounter()
        self.cpu = TernaryCPU(neural_mode=False)

    def test_alu_operations(self):
        """Test all ALU operations"""
        test_cases = [
            ("ADD", 2, 2, 1),  # 2+2=4≡1 mod 3
            ("ADD", 1, 1, 2),
            ("SUB", 2, 1, 1),
            ("SUB", 0, 1, 2),
            ("AND", 2, 1, 1),
            ("OR", 1, 2, 2),
            ("XOR", 1, 2, 0)
        ]
        
        for op, a, b, expected in test_cases:
            with self.subTest(op=op, a=a, b=b):
                result = self.alu.execute(op, a, b)
                self.assertEqual(result, expected)

    def test_register_operations(self):
        """Test register read/write operations"""
        self.registers.write(0, 2)
        self.assertEqual(self.registers.read(0), 2)
        
        # Test wrapping
        self.registers.write(10, 1)  # Should wrap to index 1
        self.assertEqual(self.registers.read(10), 1)
        self.assertEqual(self.registers.read(1), 1)

    def test_memory_operations(self):
        """Test memory load/store operations"""
        self.memory.store(0, 2)
        self.assertEqual(self.memory.load(0), 2)
        
        # Test address wrapping
        self.memory.store(30, 1)  # Should wrap with size=27
        self.assertEqual(self.memory.load(30), 1)
        self.assertEqual(self.memory.load(3), 1)  # 30 % 27 = 3

    def test_program_counter(self):
        """Test PC increment and jump operations"""
        self.assertEqual(self.pc.get(), 0)
        self.pc.increment()
        self.assertEqual(self.pc.get(), 1)
        
        self.pc.jump(10)
        self.assertEqual(self.pc.get(), 10)
        
        # Test max value
        self.pc.jump(242)
        self.pc.increment()
        self.assertEqual(self.pc.get(), 0)

    def test_instruction_packing(self):
        """Test instruction packing/unpacking"""
        test_instructions = [
            ("LOADI", 1, 2),  # Load immediate 2 into R1
            ("ADD", 2, 3, 4),  # R2 = R3 + R4
            ("HLT",)
        ]
        
        for instruction in test_instructions:
            with self.subTest(instruction=instruction):
                packed = self.cpu._pack_instruction(instruction)
                unpacked = self.cpu._unpack_instruction(packed)
                
                # For variable length instructions, compare relevant parts
                min_len = min(len(instruction), len(unpacked))
                self.assertEqual(instruction[:min_len], unpacked[:min_len])

    def test_assembly_parsing(self):
        """Test assembly code parsing"""
        assembly_code = """
            LOADI 1 2    # Load value 2 into R1
            ADD 2 1 1    # R2 = R1 + R1
            HLT          # Stop execution
        """
        
        expected = [
            ("LOADI", 1, 2),
            ("ADD", 2, 1, 1),
            ("HLT",)
        ]
        
        assembled = assemble(assembly_code.splitlines())
        self.assertEqual(assembled, expected)

    def test_cpu_execution(self):
        """Test complete CPU execution cycle"""
        program = [
            ("LOADI", 0, 2),  # R0 = 2
            ("LOADI", 1, 1),  # R1 = 1
            ("ADD", 2, 0, 1), # R2 = R0 + R1
            ("HLT",)
        ]
        
        self.cpu.load_program(program)
        self.cpu.run()
        
        # Verify registers
        self.assertEqual(self.cpu.registers.read(0), 2)
        self.assertEqual(self.cpu.registers.read(1), 1)
        self.assertEqual(self.cpu.registers.read(2), 0)  # 2 + 1 = 0 (mod 3)

    def test_neural_execution(self):
        """Test neural execution path (when enabled)"""
        neural_cpu = TernaryCPU(neural_mode=True)
        program = [
            ("LOADI", 0, 2),
            ("LOADI", 1, 1),
            ("AND", 2, 0, 1),  # Should use neural execution
            ("HLT",)
        ]
        
        neural_cpu.load_program(program)
        neural_cpu.run()
        
        # We can't predict exact neural output, but should be valid ternary
        result = neural_cpu.registers.read(2)
        self.assertIn(result, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
