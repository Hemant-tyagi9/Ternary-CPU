import sys
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any

# Import all the modules
from Security.ternary_crypto import TernaryCrypto
from Quantum.quantum_spiking import main as quantum_spiking_main
from Quantum.quantum_interface import QuantumTernaryInterface
from Quantum.qiskit_helpers import QiskitTernaryHelpers
from Quantum.entanglement_sim import main as entanglement_sim_main

def run_ternary_crypto_demo():
    """Demonstrate ternary cryptographic operations"""
    print("\n=== Ternary Cryptography Demo ===")
    crypto = TernaryCrypto("secret_key")
    
    # Encrypt a message
    message = "Hello Quantum World!"
    cipher, signature = crypto.encrypt(message)
    print(f"Original message: {message}")
    print(f"Cipher vector: {cipher}")
    print(f"Signature vector: {signature}")
    
    # Verify the message
    is_valid = crypto.verify(cipher, signature)
    print(f"Signature valid: {is_valid}")
    
    # Decrypt (simplified for demo)
    decrypted = crypto.decrypt(cipher)
    print(f"Decrypted vector: {decrypted}")

def run_quantum_spiking_demo():
    """Run the quantum spiking demonstration"""
    print("\n=== Quantum Spiking Demo ===")
    quantum_spiking_main()

def run_quantum_ternary_operations():
    """Demonstrate quantum ternary operations"""
    print("\n=== Quantum Ternary Operations Demo ===")
    qti = QuantumTernaryInterface()
    
    # Test ternary AND gate
    print("\nTesting Ternary AND Gate:")
    for a in range(3):
        for b in range(3):
            result = qti.execute_ternary_gate("AND", a, b)
            print(f"{a} AND {b} = {result}")
    
    # Test ternary OR gate
    print("\nTesting Ternary OR Gate:")
    for a in range(3):
        for b in range(3):
            result = qti.execute_ternary_gate("OR", a, b)
            print(f"{a} OR {b} = {result}")

def run_entanglement_demo():
    """Run the quantum entanglement demonstration"""
    print("\n=== Quantum Entanglement Demo ===")
    entanglement_sim_main()

def run_qiskit_helpers_demo():
    """Demonstrate Qiskit ternary helper functions"""
    print("\n=== Qiskit Ternary Helpers Demo ===")
    from qiskit import QuantumCircuit
    
    # Create a circuit with ternary registers
    num_qutrits = 2
    reg = QiskitTernaryHelpers.create_ternary_register(num_qutrits)
    qc = QuantumCircuit(reg)
    
    # Encode ternary values
    QiskitTernaryHelpers.encode_ternary(qc, reg, 1, 0)  # First qutrit = 1
    QiskitTernaryHelpers.encode_ternary(qc, reg, 2, 1)  # Second qutrit = 2
    
    # Perform ternary AND operation
    QiskitTernaryHelpers.ternary_and(qc, 0, 1, 0, reg)  # Result in position 0
    
    print("\nCircuit with ternary operations:")
    print(qc)

def main():
    """Main function to run all demonstrations"""
    print("=== Advanced CPU Demonstrations ===")
    print("1. Ternary Cryptography")
    print("2. Quantum Spiking")
    print("3. Quantum Ternary Operations")
    print("4. Quantum Entanglement")
    print("5. Qiskit Ternary Helpers")
    print("6. Run All Demos")
    print("0. Exit")
    
    while True:
        choice = input("\nSelect a demo to run (0-6): ")
        
        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            run_ternary_crypto_demo()
        elif choice == "2":
            run_quantum_spiking_demo()
        elif choice == "3":
            run_quantum_ternary_operations()
        elif choice == "4":
            run_entanglement_demo()
        elif choice == "5":
            run_qiskit_helpers_demo()
        elif choice == "6":
            run_ternary_crypto_demo()
            run_quantum_spiking_demo()
            run_quantum_ternary_operations()
            run_entanglement_demo()
            run_qiskit_helpers_demo()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = Path("results/quantum")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)
