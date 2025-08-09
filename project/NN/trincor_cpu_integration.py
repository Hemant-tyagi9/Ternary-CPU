#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import neural processing unit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .nn_models import TernaryNeuralCPU, create_neural_cpu


class ProcessingMode(Enum):
    """CPU Processing modes"""
    TRADITIONAL = "traditional"
    NEURAL = "neural"
    HYBRID = "hybrid"
    AUTO = "auto"


class InstructionType(Enum):
    """Types of ternary instructions"""
    LOGIC = "logic"        # AND, OR, XOR, NAND, NOR
    ARITHMETIC = "arithmetic"  # ADD, SUB, MUL, DIV
    CONTROL = "control"    # JMP, CALL, RET
    MEMORY = "memory"      # LOAD, STORE
    NEURAL = "neural"      # Neural-specific operations


@dataclass
class Instruction:
    """Ternary CPU instruction representation"""
    opcode: str
    operand_a: int
    operand_b: int
    destination: Optional[int] = None
    instruction_type: InstructionType = InstructionType.LOGIC
    processing_mode: ProcessingMode = ProcessingMode.AUTO


class TernaryRegisterFile:
    """Simple ternary register file simulation"""
    
    def __init__(self, num_registers: int = 16):
        self.registers = [0] * num_registers
        self.num_registers = num_registers
    
    def read(self, register_id: int) -> int:
        if 0 <= register_id < self.num_registers:
            return self.registers[register_id]
        return 0
    
    def write(self, register_id: int, value: int):
        if 0 <= register_id < self.num_registers:
            self.registers[register_id] = max(0, min(2, value))  # Clamp to ternary range
    
    def get_state(self) -> List[int]:
        return self.registers.copy()


class TraditionalProcessor:
    """Traditional ternary logic processor for comparison"""
    
    @staticmethod
    def execute_and(a: int, b: int) -> int:
        """Traditional ternary AND"""
        if a == 0 or b == 0:
            return 0
        elif a == 2 and b == 2:
            return 2
        else:
            return 1
    
    @staticmethod
    def execute_or(a: int, b: int) -> int:
        """Traditional ternary OR"""
        return max(a, b)
    
    @staticmethod
    def execute_xor(a: int, b: int) -> int:
        """Traditional ternary XOR"""
        return (a + b) % 3 if a != b else 0
    
    @staticmethod
    def execute_add(a: int, b: int) -> int:
        """Traditional ternary ADD"""
        return (a + b) % 3
    
    @staticmethod
    def execute_sub(a: int, b: int) -> int:
        """Traditional ternary SUB"""
        return (a - b) % 3
    
    def execute_operation(self, operation: str, a: int, b: int) -> int:
        """Execute operation using traditional logic"""
        operations = {
            "AND": self.execute_and,
            "OR": self.execute_or,
            "XOR": self.execute_xor,
            "ADD": self.execute_add,
            "SUB": self.execute_sub
        }
        
        if operation in operations:
            return operations[operation](a, b)
        else:
            raise ValueError(f"Unsupported traditional operation: {operation}")


class TrinCoreHybridCPU:
    """
    Main TrinCore CPU with hybrid neural-traditional processing
    """
    
    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.HYBRID):
        self.processing_mode = processing_mode
        self.registers = TernaryRegisterFile(16)
        self.program_counter = 0
        self.cycle_count = 0
        
        # Processing units
        self.neural_unit = create_neural_cpu(auto_optimize=True, cache_models=True)
        self.traditional_unit = TraditionalProcessor()
        
        # Performance tracking
        self.neural_operations = 0
        self.traditional_operations = 0
        self.total_operations = 0
        self.performance_history = []
        
        # Operation routing rules
        self.neural_preferred = {"XOR", "NAND", "NOR"}  # Complex operations
        self.traditional_preferred = {"ADD", "SUB"}      # Simple arithmetic
        self.either_ok = {"AND", "OR"}                   # Either works well
        
        print(f"TrinCore CPU initialized in {processing_mode.value} mode")
    
    def _decide_processing_unit(self, operation: str) -> str:
        """Decide which processing unit to use for an operation"""
        if self.processing_mode == ProcessingMode.NEURAL:
            return "neural"
        elif self.processing_mode == ProcessingMode.TRADITIONAL:
            return "traditional"
        elif self.processing_mode == ProcessingMode.AUTO:
            # Use performance history to decide
            if operation in self.neural_preferred:
                return "neural"
            elif operation in self.traditional_preferred:
                return "traditional"
            else:
                # Use the faster one based on recent performance
                return self._get_faster_unit(operation)
        else:  # HYBRID mode
            # Intelligent routing based on operation type and current load
            if operation in self.neural_preferred:
                return "neural"
            elif operation in self.traditional_preferred:
                return "traditional"
            else:
                # Balance load between units
                neural_load = self.neural_operations / max(1, self.total_operations)
                return "neural" if neural_load < 0.6 else "traditional"
    
    def _get_faster_unit(self, operation: str) -> str:
        """Get the historically faster unit for an operation"""
        # In a real implementation, this would use actual performance data
        # For demo, use simple heuristics
        if operation in {"XOR", "NAND", "NOR"}:
            return "neural"  # Neural usually better for complex logic
        else:
            return "traditional"  # Traditional faster for simple ops
    
    def execute_instruction(self, instruction: Instruction) -> Tuple[int, Dict[str, Any]]:
        """Execute a single instruction"""
        start_time = time.perf_counter()
        
        # Read operands from registers or use direct values
        operand_a = instruction.operand_a
        operand_b = instruction.operand_b
        
        # Decide processing unit
        processing_unit = self._decide_processing_unit(instruction.opcode)
        
        # Execute operation
        try:
            if processing_unit == "neural":
                result = self.neural_unit.execute_operation(
                    instruction.opcode, operand_a, operand_b
                )
                self.neural_operations += 1
            else:
                result = self.traditional_unit.execute_operation(
                    instruction.opcode, operand_a, operand_b
                )
                self.traditional_operations += 1
            
            self.total_operations += 1
            execution_time = time.perf_counter() - start_time
            
            # Write result to destination register if specified
            if instruction.destination is not None:
                self.registers.write(instruction.destination, result)
            
            # Update performance tracking
            perf_data = {
                'operation': instruction.opcode,
                'processing_unit': processing_unit,
                'execution_time': execution_time,
                'cycle': self.cycle_count,
                'result': result
            }
            self.performance_history.append(perf_data)
            
            self.cycle_count += 1
            
            return result, perf_data
            
        except Exception as e:
            print(f"Execution error: {e}")
            return 0, {'error': str(e)}
    
    def execute_program(self, instructions: List[Instruction]) -> Dict[str, Any]:
        """Execute a complete program"""
        print(f"Executing program with {len(instructions)} instructions...")
        
        start_time = time.time()
        results = []
        errors = 0
        
        for i, instruction in enumerate(instructions):
            try:
                result, perf_data = self.execute_instruction(instruction)
                results.append(result)
                
                if 'error' in perf_data:
                    errors += 1
                    
            except Exception as e:
                print(f"Error executing instruction {i}: {e}")
                results.append(0)
                errors += 1
        
        total_time = time.time() - start_time
        
        return {
            'results': results,
            'total_time': total_time,
            'instructions_per_second': len(instructions) / total_time,
            'neural_operations': self.neural_operations,
            'traditional_operations': self.traditional_operations,
            'total_operations': self.total_operations,
            'errors': errors,
            'final_register_state': self.registers.get_state()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {}
        
        neural_times = [p['execution_time'] for p in self.performance_history 
                       if p['processing_unit'] == 'neural']
        traditional_times = [p['execution_time'] for p in self.performance_history 
                           if p['processing_unit'] == 'traditional']
        
        return {
            'total_operations': self.total_operations,
            'neural_operations': self.neural_operations,
            'traditional_operations': self.traditional_operations,
            'neural_percentage': (self.neural_operations / max(1, self.total_operations)) * 100,
            'avg_neural_time': np.mean(neural_times) if neural_times else 0,
            'avg_traditional_time': np.mean(traditional_times) if traditional_times else 0,
            'operations_by_type': self._get_operation_breakdown(),
            'performance_trend': self._analyze_performance_trend()
        }
    
    def _get_operation_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get breakdown of operations by type and processing unit"""
        breakdown = {}
        
        for perf in self.performance_history:
            op = perf['operation']
            unit = perf['processing_unit']
            
            if op not in breakdown:
                breakdown[op] = {'neural': 0, 'traditional': 0}
            
            breakdown[op][unit] += 1
        
        return breakdown
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        if len(self.performance_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Get recent performance (last 50% of operations)
        recent_ops = self.performance_history[len(self.performance_history)//2:]
        
        neural_recent = [p for p in recent_ops if p['processing_unit'] == 'neural']
        traditional_recent = [p for p in recent_ops if p['processing_unit'] == 'traditional']
        
        return {
            'recent_neural_ratio': len(neural_recent) / len(recent_ops),
            'recent_avg_neural_time': np.mean([p['execution_time'] for p in neural_recent]) if neural_recent else 0,
            'recent_avg_traditional_time': np.mean([p['execution_time'] for p in traditional_recent]) if traditional_recent else 0,
            'trend': 'neural_increasing' if len(neural_recent) > len(traditional_recent) else 'traditional_increasing'
        }
    
    def optimize_performance(self):
        """Optimize CPU performance based on usage patterns"""
        print("Optimizing CPU performance...")
        
        # Analyze most used operations
        op_counts = {}
        for perf in self.performance_history:
            op = perf['operation']
            op_counts[op] = op_counts.get(op, 0) + 1
        
        # Get top operations
        top_operations = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        optimization_targets = [op for op, count in top_operations if count > 10]
        
        # Optimize neural models for frequently used operations
        if optimization_targets:
            print(f"Optimizing neural models for: {', '.join(optimization_targets)}")
            self.neural_unit.optimize_models(optimization_targets)
        
        # Update routing preferences based on performance
        self._update_routing_preferences()
    
    def _update_routing_preferences(self):
        """Update operation routing based on observed performance"""
        op_performance = {}
        
        for perf in self.performance_history[-100:]:  # Last 100 operations
            op = perf['operation']
            unit = perf['processing_unit']
            time = perf['execution_time']
            
            if op not in op_performance:
                op_performance[op] = {'neural': [], 'traditional': []}
            
            op_performance[op][unit].append(time)
        
        # Update preferences based on average times
        for op, times in op_performance.items():
            if times['neural'] and times['traditional']:
                neural_avg = np.mean(times['neural'])
                traditional_avg = np.mean(times['traditional'])
                
                if neural_avg < traditional_avg * 0.8:  # Neural significantly faster
                    self.neural_preferred.add(op)
                    self.traditional_preferred.discard(op)
                elif traditional_avg < neural_avg * 0.8:  # Traditional significantly faster
                    self.traditional_preferred.add(op)
                    self.neural_preferred.discard(op)
                else:  # Similar performance
                    self.either_ok.add(op)


def create_sample_program() -> List[Instruction]:
    """Create a sample ternary program for demonstration"""
    return [
        # Test basic logic operations
        Instruction("AND", 2, 1, destination=0),
        Instruction("OR", 1, 2, destination=1),
        Instruction("XOR", 2, 1, destination=2),
        
        # Test arithmetic
        Instruction("ADD", 1, 1, destination=3),
        Instruction("SUB", 2, 1, destination=4),
        
        # Test complex operations
        Instruction("NAND", 2, 2, destination=5),
        Instruction("NOR", 1, 0, destination=6),
        
        # Test pattern with repeated operations
        Instruction("AND", 1, 2, destination=7),
        Instruction("AND", 2, 1, destination=8),
        Instruction("XOR", 0, 2, destination=9),
        Instruction("XOR", 1, 1, destination=10),
        
        # More complex sequence
        Instruction("OR", 2, 0, destination=11),
        Instruction("ADD", 2, 2, destination=12),
        Instruction("SUB", 1, 2, destination=13),
    ]


def demonstrate_cpu_modes():
    """Demonstrate different CPU processing modes"""
    print("="*60)
    print("TRINCORE CPU MODE COMPARISON")
    print("="*60)
    
    sample_program = create_sample_program()
    modes = [ProcessingMode.TRADITIONAL, ProcessingMode.NEURAL, ProcessingMode.HYBRID]
    
    results = {}
    
    for mode in modes:
        print(f"\nTesting {mode.value.upper()} mode:")
        print("-" * 40)
        
        cpu = TrinCoreHybridCPU(processing_mode=mode)
        program_results = cpu.execute_program(sample_program)
        perf_summary = cpu.get_performance_summary()
        
        results[mode.value] = {
            'execution_time': program_results['total_time'],
            'instructions_per_second': program_results['instructions_per_second'],
            'neural_percentage': perf_summary.get('neural_percentage', 0),
            'errors': program_results['errors']
        }
        
        print(f"Execution time: {program_results['total_time']*1000:.2f} ms")
        print(f"Instructions per second: {program_results['instructions_per_second']:.0f}")
        print(f"Neural operations: {perf_summary.get('neural_percentage', 0):.1f}%")
        print(f"Errors: {program_results['errors']}")
    
    # Compare results
    print(f"\n{'Mode':<12} {'Time (ms)':<12} {'IPS':<12} {'Neural %':<12} {'Errors'}")
    print("-" * 60)
    
    for mode, result in results.items():
        print(f"{mode.capitalize():<12} {result['execution_time']*1000:<12.2f} "
              f"{result['instructions_per_second']:<12.0f} {result['neural_percentage']:<12.1f} "
              f"{result['errors']:<12}")


def demonstrate_adaptive_optimization():
    """Demonstrate adaptive performance optimization"""
    print("\n" + "="*60)
    print("ADAPTIVE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    cpu = TrinCoreHybridCPU(processing_mode=ProcessingMode.AUTO)
    
    # Create a workload that heavily uses certain operations
    heavy_workload = []
    
    # Phase 1: Heavy XOR usage
    for _ in range(50):
        heavy_workload.extend([
            Instruction("XOR", np.random.randint(0, 3), np.random.randint(0, 3)),
            Instruction("AND", np.random.randint(0, 3), np.random.randint(0, 3)),
        ])
    
    # Phase 2: Heavy ADD usage
    for _ in range(50):
        heavy_workload.extend([
            Instruction("ADD", np.random.randint(0, 3), np.random.randint(0, 3)),
            Instruction("SUB", np.random.randint(0, 3), np.random.randint(0, 3)),
        ])
    
    print("Phase 1: Initial execution")
    phase1_results = cpu.execute_program(heavy_workload[:100])
    phase1_perf = cpu.get_performance_summary()
    
    print(f"Initial neural usage: {phase1_perf.get('neural_percentage', 0):.1f}%")
    print(f"Initial performance: {phase1_results['instructions_per_second']:.0f} IPS")
    
    print("\nOptimizing based on usage patterns...")
    cpu.optimize_performance()
    
    print("\nPhase 2: Post-optimization execution")
    phase2_results = cpu.execute_program(heavy_workload[100:])
    phase2_perf = cpu.get_performance_summary()
    
    print(f"Optimized neural usage: {phase2_perf.get('neural_percentage', 0):.1f}%")
    print(f"Optimized performance: {phase2_results['instructions_per_second']:.0f} IPS")
    
    improvement = ((phase2_results['instructions_per_second'] - 
                   phase1_results['instructions_per_second']) / 
                   phase1_results['instructions_per_second']) * 100
    
    print(f"Performance improvement: {improvement:+.1f}%")


def main():
    """Main demonstration function"""
    print("TrinCore Hybrid Neural-Traditional CPU Demonstration")
    
    try:
        demonstrate_cpu_modes()
        demonstrate_adaptive_optimization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("TrinCore CPU successfully demonstrated hybrid neural-traditional processing!")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
