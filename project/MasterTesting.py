import unittest
import numpy as np
from Neuromorphic.spiking_network import SpikingNeuralNetwork
from Integration.cpu_extend import TernaryCPU

class TestNeuromorphicCPU(unittest.TestCase):
    
    def setUp(self):
        self.cpu = TernaryCPU(neural_mode=True)
        self.spiking_net = SpikingNeuralNetwork()
    
    def test_spiking_network(self):
        """Test basic spiking network functionality"""
        inputs = [1, 2]  # Ternary inputs
        outputs = self.spiking_net.spike(inputs)
        self.assertTrue(len(outputs) > 0)
    
    def test_hybrid_execution(self):
        """Test CPU with neural acceleration"""
        program = [
            ("LOADI", 0, 1),
            ("LOADI", 1, 2),
            ("AND", 2, 0, 1),
            ("HLT",)
        ]
        self.cpu.load_program(program)
        self.cpu.run()
        self.assertEqual(self.cpu.registers.read(2), 1)  # 1 AND 2 = 1
        
    def test_quantum_interface(self):
        """Test quantum-ternary gate simulation"""
        from Quantum.quantum_interface import QuantumTernaryInterface
        qti = QuantumTernaryInterface()
        result = qti.execute_ternary_gate("AND", 1, 2)
        self.assertIn(result, [0, 1, 2])

if __name__ == "__main__":
    unittest.main()
