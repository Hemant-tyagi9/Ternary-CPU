import os
import sys
from datetime import datetime
from Cpu_components.ternary_gates import *
from Integration.cpu_extend import TernaryCPU
from NN.nn_loader import load_models_from_dir
from Neuromorphic.spiking_neurons import SpikingNeuralNetwork

def print_banner():
    print("""
    ╔════════════════════════════════════════════════╗
    ║    NEUROMORPHIC TERNARY CPU SIMULATION         ║
    ║  Hybrid Traditional + Neural Architecture      ║
    ╚════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    print(f"[BOOT] Starting at {datetime.now()}")
    
    # Initialize components
    cpu = TernaryCPU(neural_mode=True)
    spiking_net = SpikingNeuralNetwork()
    
    # Load neural models
    print("[INFO] Loading neural models...")
    nn_models = load_models_from_dir("models")
    
    # Sample program
    program = [
        ("LOADI", 0, 1),   # Load 1 into register 0
        ("LOADI", 1, 2),   # Load 2 into register 1
        ("ADD", 2, 0, 1),  # Add reg0 + reg1 -> reg2
        ("AND", 3, 0, 1),  # AND reg0 & reg1 -> reg3
        ("HLT",)           # Halt
    ]
    
    # Execute
    cpu.load_program(program)
    print("[INFO] Starting execution...")
    cpu.run()
    
    # Display results
    print("\n[RESULTS]")
    for i in range(4):
        print(f"R{i} = {cpu.registers.read(i)}")
    
    print("\n[STATS]")
    print(f"Neural ops: {cpu.operation_stats.get('neural', 0)}")
    print(f"Traditional ops: {sum(cpu.operation_stats.values()) - cpu.operation_stats.get('neural', 0)}")

if __name__ == "__main__":
    main()
