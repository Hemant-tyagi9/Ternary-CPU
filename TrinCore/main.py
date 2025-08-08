from trincore_applications.config import TrinCoreConfig
from trincore_applications.logging import logger
import argparse
import time

# Your existing imports
from CPU_Components.alu import TernaryALU
from CPU_Components.ternary_memory import TernaryMemory
from CPU_Components.register_sets import RegisterSet
from CPU_Components.program_counter import ProgramCounter
from NN.nn_integration import NeuralIntegration
from Integration.cpu_extend import TernaryCPU

# Neuromorphic imports
from Neuromorphic.event_driven_alu import FullyNeuromorphicTernaryCPU

def initialize_cpu(mode="traditional"):
    """Initialize CPU in different modes with config support"""
    config = TrinCoreConfig()
    
    logger.info(f"Initializing CPU in {mode} mode")
    
    if mode == "traditional":
        memory = TernaryMemory(size=config.get("memory.size", 729))
        registers = RegisterSet(num_registers=config.get("registers.count", 9))
        pc = ProgramCounter()
        alu = TernaryALU()
        return TernaryCPU(memory=memory, alu=alu, registers=registers, pc=pc)
    
    elif mode == "neural":
        memory = TernaryMemory(size=config.get("memory.size", 729))
        registers = RegisterSet(num_registers=config.get("registers.count", 9))
        pc = ProgramCounter()
        neural_alu = NeuralIntegration()
        ops = config.get("nn.default_operations", ["ADD", "SUB", "AND", "OR", "XOR"])
        neural_alu.train_models(ops)
        return TernaryCPU(memory=memory, alu=neural_alu, registers=registers, pc=pc)
    
    elif mode == "neuromorphic":
        return FullyNeuromorphicTernaryCPU()
    
    else:
        logger.error(f"Unknown CPU mode: {mode}")
        raise ValueError(f"Unknown mode: {mode}")

def run_demo_program(cpu, mode="traditional"):
    """Run demo program with enhanced logging"""
    logger.info(f"Starting demo program in {mode} mode")
    
    # Define test program based on mode
    if mode == "neuromorphic":
        program = [
            ("LOAD", 0, 10),
            ("LOAD", 1, 20),
            ("ADD", 2, 0, 1),
            ("STORE", 2, 30),
            ("LOAD", 3, 30),
            ("XOR", 4, 2, 3),
            ("AND", 5, 2, 1),
        ]
        
        # Initialize memory
        cpu.neuromorphic_memory[10].data = 1
        cpu.neuromorphic_memory[20].data = 2
        
        cpu.start_neuromorphic_mode()
        cpu.run_neuromorphic_program(program)
        
        # Show results
        for reg in range(6):
            logger.info(f"Register R{reg} = {cpu.registers.read(reg)}")
        
        metrics = cpu.get_complete_metrics()
        logger.info("Neuromorphic metrics:\n" + "\n".join(
            f"{k:25}: {v}" for k,v in metrics.items()
        ))
        
        cpu.stop_neuromorphic_processing()
    else:
        program = [
            ("LOADI", 0, 1),
            ("LOADI", 1, 2),
            ("ADD", 2, 0, 1),
            ("PRINT", 2),
            ("HLT",)
        ]
        cpu.load_program(program)
        cpu.run()

def compare_all_modes():
    """Enhanced mode comparison with logging"""
    config = TrinCoreConfig()
    modes = config.get("system.available_modes", ["traditional", "neural", "neuromorphic"])
    results = {}
    
    logger.info("Starting comprehensive mode comparison")
    
    for mode in modes:
        try:
            logger.info(f"\n{'='*20} {mode.upper()} MODE {'='*20}")
            start_time = time.time()
            
            cpu = initialize_cpu(mode)
            run_demo_program(cpu, mode)
            
            exec_time = time.time() - start_time
            results[mode] = {"execution_time": exec_time}
            
            if mode == "neuromorphic":
                metrics = cpu.get_complete_metrics()
                results[mode].update({
                    "events_processed": metrics.get("events_processed", 0),
                    "parallel_ops": metrics.get("parallel_operations", 0)
                })
            
            logger.info(f"{mode.title()} mode completed in {exec_time:.4f}s")
        except Exception as e:
            logger.error(f"Error in {mode} mode: {str(e)}")
            results[mode] = {"error": str(e)}
    
    # Print comparison summary
    logger.info("\n📊 FINAL COMPARISON SUMMARY:")
    for mode, data in results.items():
        logger.info(f"{mode.upper():15}")
        for key, value in data.items():
            logger.info(f"  {key:20}: {value}")

def main():
    """Enhanced main function with config and logging support"""
    parser = argparse.ArgumentParser(
        description="TrinCore Ternary CPU Simulator with Enhanced Features"
    )
    parser.add_argument('--mode', choices=['traditional', 'neural', 'neuromorphic', 'compare'], 
                       default='traditional', help='CPU execution mode')
    parser.add_argument('--demo', action='store_true', help='Run demo program')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Load custom config if specified
    if args.config:
        TrinCoreConfig().load(args.config)
    
    logger.info("🚀 Starting TrinCore Ternary CPU Simulator")
    logger.info(f"System mode: {TrinCoreConfig().get('system.mode', 'balanced')}")
    logger.info(f"Log level: {TrinCoreConfig().get('system.log_level', 'INFO')}")
    
    if args.mode == 'compare':
        compare_all_modes()
    else:
        cpu = initialize_cpu(args.mode)
        
        if args.demo:
            run_demo_program(cpu, args.mode)
        
        if args.benchmark and args.mode == 'neuromorphic':
            logger.info("Starting neuromorphic benchmark...")
            benchmark_neuromorphic_performance(cpu)

if __name__ == "__main__":
    main()
