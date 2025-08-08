from NN.nn_integration import NeuralIntegration
from CPU_Components.alu import TernaryALU
from CPU_Components.ternary_memory import TernaryMemory
from CPU_Components.register_sets import RegisterSet
from CPU_Components.program_counter import ProgramCounter

class NeuromorphicTernaryCPU:
    """Hybrid traditional/neural ternary CPU implementation"""
    
    def __init__(self, neural_ratio=0.5):
        self.memory = TernaryMemory()
        self.registers = RegisterSet()
        self.pc = ProgramCounter()
        self.neural_alu = NeuralIntegration()
        self.traditional_alu = TernaryALU()
        self.neural_ratio = neural_ratio
        
        # Train neural models for common operations
        self.neural_alu.train_models(["ADD", "SUB", "AND", "OR", "XOR"])
    
    def execute(self, opcode, a, b):
        """Execute operation with probabilistic neural/traditional ALU selection"""
        use_neural = np.random.random() < self.neural_ratio
        if use_neural and self.neural_alu.has_model(opcode):
            return self.neural_alu.execute_operation(opcode, a, b)
        else:
            return self.traditional_alu.execute(opcode, a, b)
    
    def run_cycle(self):
        """Execute one CPU cycle"""
        instruction = self.fetch()
        self.decode_execute(instruction)
        self.pc.increment()
    
