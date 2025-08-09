from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate, CCXGate
from qiskit.quantum_info import Statevector
import numpy as np

class QiskitTernaryHelpers:
    """Helper functions for ternary-quantum operations using Qiskit"""
    
    @staticmethod
    def create_ternary_register(num_qutrits: int) -> QuantumRegister:
        """Create a register that can represent ternary values"""
        # Using 2 qubits per qutrit (00=0, 01=1, 11=2)
        return QuantumRegister(2 * num_qutrits, 'qutrit')
    
    @staticmethod
    def encode_ternary(qc: QuantumCircuit, reg: QuantumRegister, value: int, pos: int):
        """Encode a ternary value onto qubits"""
        if value == 1:
            qc.x(reg[2*pos])
        elif value == 2:
            qc.x(reg[2*pos])
            qc.x(reg[2*pos+1])
    
    @staticmethod
    def ternary_and(qc: QuantumCircuit, a: int, b: int, target: int, reg: QuantumRegister):
        """Implement ternary AND using quantum gates"""
        # Using Toffoli gate for AND operation
        qc.append(CCXGate(), [reg[2*a], reg[2*b], reg[2*target]])
    
    @staticmethod
    def measure_ternary(qc: QuantumCircuit, reg: QuantumRegister, pos: int, cbit: int):
        """Measure a ternary value (2 qubits -> 1 ternary digit)"""
        qc.measure(reg[2*pos], cbit)
        qc.measure(reg[2*pos+1], cbit+1)
    
    @staticmethod
    def decode_ternary(counts: dict) -> dict:
        """Convert binary measurements to ternary counts"""
        ternary_counts = {}
        for binary, count in counts.items():
            # Reverse string since Qiskit uses little-endian
            rev = binary[::-1]
            ternary = ''
            for i in range(0, len(rev), 2):
                bits = rev[i:i+2]
                if bits == '00':
                    ternary += '0'
                elif bits == '01':
                    ternary += '1'
                elif bits == '10':
                    ternary += '1'  # Alternative encoding
                elif bits == '11':
                    ternary += '2'
            ternary_counts[ternary] = ternary_counts.get(ternary, 0) + count
        return ternary_counts
    
    @staticmethod
    def ternary_superposition(qc: QuantumCircuit, reg: QuantumRegister, pos: int):
        """Create superposition state representing ternary superposition"""
        qc.h(reg[2*pos])
        qc.h(reg[2*pos+1])
        # Adjust phases for equal superposition
        qc.p(2*np.pi/3, reg[2*pos])
        qc.p(4*np.pi/3, reg[2*pos+1])
