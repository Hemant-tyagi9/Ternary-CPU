import time
from typing import Dict, Any
from CPU_Components.ternary_gates import *
from typing import List

class HardwareSimulator:
    """Simulates hardware-level behavior of ternary components"""
    
    def __init__(self):
        self.gate_delays = {
            'AND': 1.5,
            'OR': 1.5,
            'XOR': 2.0,
            'NAND': 1.7,
            'NOR': 1.7,
            'NOT': 1.0
        }
        self.voltage_levels = {0: 0.0, 1: 1.5, 2: 3.0}
    
    def simulate_gate(self, gate_type: str, inputs: List[int]) -> Dict[str, Any]:
        """Simulate gate operation with timing and voltage characteristics"""
        start_time = time.perf_counter()
        
        # Get gate function
        gate_fn = {
            'AND': ternary_and,
            'OR': ternary_or,
            'XOR': ternary_xor,
            'NAND': ternary_nand,
            'NOR': ternary_nor,
            'NOT': ternary_not
        }.get(gate_type, lambda *x: 0)
        
        # Execute gate operation
        result = gate_fn(*inputs)
        
        # Simulate propagation delay
        time.sleep(self.gate_delays.get(gate_type, 1.0) * 1e-9)
        
        return {
            'result': result,
            'voltage': self.voltage_levels[result],
            'latency': time.perf_counter() - start_time
        }
    
    def simulate_bus(self, data: List[int]) -> Dict[str, Any]:
        """Simulate data bus transmission"""
        start_time = time.perf_counter()
        voltages = [self.voltage_levels[d] for d in data]
        time.sleep(2.0 * 1e-9)  # Bus propagation delay
        return {
            'voltages': voltages,
            'latency': time.perf_counter() - start_time
        }
