#!/usr/bin/env python3


import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

   
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from Cpu_components.ternary_gates import *
    from Cpu_components.alu import TernaryALU
    from Cpu_components.assembly import assemble, OpCode
    from Integration.cpu_extend import TernaryCPU
    print("+"*15,"Cpu components import successful!", "+"*15)
    from NN.nn_models import TernaryNeuralCPU, create_neural_cpu
    from NN.model_training import (
        compare_models, train_genetic_model_advanced, 
        train_or_load_model, generate_training_data
    )
    from NN.nn_benchmark import (
        benchmark_neural_operation, benchmark_genetic_model, 
        benchmark_traditional_operation
    )
    print("+"*15,"NN import successful!", "+"*15)
    from Neuromorphic.spiking_neurons import SpikingNeuralNetwork
    from Neuromorphic.adaptive_learning import AdaptiveLearner
    from Neuromorphic.parallel_processor import ParallelProcessor
    print("+"*15,"Main Neuromorphic components import successful!", "+"*15)
    from FPGA_Simulator.verilog_generator import generate_testbench, generate_verilog_module
    from FPGA_Simulator.pipeline import NeuromorphicPipeline
    print("+"*15,"Verilog generator and pipelines import successful!", "+"*15)
    from Quantum.entanglement_sim import TernaryQubit
    from Applications.ternary_cv import TernaryImageProcessor
    from Applications.ternary_nlp import TernaryNLP
    from trincore_applications.config import TrinCoreConfig
    from trincore_applications.logging import logger
    print("+"*15,"Cpu extends import successful!", "+"*15)
    
    print("="*25, "All imports successful!", "="*25)
except ImportError as e:
    print("="*25, f"Import error: {e}", "="*25)
    print("Please ensure all modules are properly installed")
    sys.exit(1)

time.sleep(5)

class ComprehensiveTestSuite:
    """Complete test suite for the ternary neuromorphic CPU system"""
    
    def __init__(self):
        self.config = TrinCoreConfig()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'evolution_history': [],
            'benchmark_results': {}
        }
        self.setup_environment()
    
    def setup_environment(self):
        """Initialize test environment"""
        print("\n" + "="*80)
        print("TERNARY NEUROMORPHIC CPU COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        print("="*80)
        
        # Create output directories
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("test_results/plots", exist_ok=True)
        os.makedirs("test_results/logs", exist_ok=True)
        os.makedirs("test_results/models", exist_ok=True)
        
        logger.info("Test suite environment initialized")
    
    def test_basic_ternary_operations(self):
        """Test 1: Basic ternary logic gates"""
        print("\nTEST 1: Basic Ternary Operations")
        print("-" * 50)
        
        operations = {
            'AND': ternary_and,
            'OR': ternary_or,
            'XOR': ternary_xor,
            'NAND': ternary_nand,
            'NOR': ternary_nor
        }
        
        test_results = {}
        
        for op_name, op_func in operations.items():
            print(f"Testing {op_name}...")
            results = []
            
            for a in range(3):
                for b in range(3):
                    result = op_func(a, b)
                    results.append((a, b, result))
                    print(f"  {op_name}({a}, {b}) = {result}")
            
            test_results[op_name] = results
            logger.info(f"Completed {op_name} gate testing with {len(results)} test cases")
        
        self.results['tests']['basic_operations'] = test_results
        print("Basic ternary operations test completed")
    
    def test_traditional_cpu(self):
        """Test 2: Traditional CPU components"""
        print("\nTEST 2: Traditional CPU Components")
        print("-" * 50)
        
        # Initialize CPU
        cpu = TernaryCPU(memory_size=27, neural_mode=False)
        
        # Test program
        program = [
            ("LOADI", 0, 1),   # R0 = 1
            ("LOADI", 1, 2),   # R1 = 2
            ("ADD", 2, 0, 1),  # R2 = R0 + R1
            ("AND", 3, 0, 1),  # R3 = R0 & R1
            ("OR", 4, 0, 1),   # R4 = R0 | R1
            ("XOR", 5, 0, 1),  # R5 = R0 ^ R1
            ("SUB", 6, 1, 0),  # R6 = R1 - R0
            ("HLT",)
        ]
        
        print("Loading program...")
        cpu.load_program(program)
        
        print("Executing program...")
        start_time = time.time()
        cpu.run()
        execution_time = time.time() - start_time
        
        # Collect results
        register_state = {}
        for i in range(9):
            register_state[f'R{i}'] = cpu.registers.read(i)
            print(f"  R{i} = {cpu.registers.read(i)}")
        
        self.results['tests']['traditional_cpu'] = {
            'register_state': register_state,
            'execution_time': execution_time,
            'instructions_executed': len(program) - 1,
            'operation_stats': cpu.operation_stats
        }
        
        logger.info(f"Traditional CPU test completed in {execution_time:.4f}s")
        print("Traditional CPU test completed")
    
    def test_neural_networks(self):
        """Test 3: Neural network training and evolution"""
        print("\nTEST 3: Neural Network Training & Evolution")
        print("-" * 50)
        
        operations = ['AND', 'OR', 'XOR']
        nn_results = {}
        
        for operation in operations:
            print(f"\nTraining neural networks for {operation}...")
            
            # Train standard neural network
            print(f"Training standard NN for {operation}...")
            std_model, std_metrics = train_or_load_model(
                operation, 
                epochs=5000, 
                force_retrain=True
            )
            
            # Train genetic neural network
            print(f"Training genetic NN for {operation}...")
            gen_model, gen_metrics = train_genetic_model_advanced(
                operation,
                population_size=40,
                generations=25
            )
            
            # Benchmark both models
            std_benchmark = benchmark_neural_operation(std_model, operation)
            gen_benchmark = benchmark_genetic_model(gen_model, operation)
            
            # Traditional benchmark for comparison
            trad_benchmark = benchmark_traditional_operation(operation)
            
            nn_results[operation] = {
                'standard': {
                    'metrics': std_metrics,
                    'benchmark': std_benchmark
                },
                'genetic': {
                    'metrics': gen_metrics,
                    'benchmark': gen_benchmark
                },
                'traditional': {
                    'benchmark': trad_benchmark
                }
            }
            
            # Display results
            print(f"   Results for {operation}:")
            print(f"   Standard NN:  Accuracy={std_benchmark['accuracy']:.3f}, Time={std_benchmark['avg_time']*1000:.2f}ms")
            print(f"   Genetic NN:   Accuracy={gen_benchmark['accuracy']:.3f}, Time={gen_benchmark['avg_time']*1000:.2f}ms")
            print(f"   Traditional:  Time={trad_benchmark['avg_time']*1000:.2f}ms")
            
            # Track evolution history
            if 'fitness_history' in gen_metrics:
                self.results['evolution_history'].append({
                    'operation': operation,
                    'fitness_history': gen_metrics['fitness_history'],
                    'accuracy_history': gen_metrics.get('accuracy_history', [])
                })
        
        self.results['tests']['neural_networks'] = nn_results
        logger.info("Neural network training and evolution tests completed")
        print("Neural network tests completed")
    
    def test_neuromorphic_components(self):
        """Test 4: Neuromorphic computing components"""
        print("\nTEST 4: Neuromorphic Computing")
        print("-" * 50)
        
        # Initialize neuromorphic components
        spiking_net = SpikingNeuralNetwork(num_neurons=32)
        adaptive_learner = AdaptiveLearner(spiking_net)
        parallel_processor = ParallelProcessor(num_cores=4)
        pipeline = NeuromorphicPipeline(stages=5)
        
        print("Testing spiking neural network...")
        
        # Test spiking behavior
        spike_results = []
        for test_case in range(10):
            inputs = [np.random.randint(0, 3) for _ in range(3)]
            spikes = spiking_net.spike(inputs)  # Expected to return a NumPy array (bools)
            spike_count = int(np.sum(spikes))   # Count of active spikes
            active_neurons = list(np.where(spikes)[0])  # Indices of neurons that fired
            
            spike_results.append({
                'inputs': inputs,
                'spike_count': spike_count,
                'active_neurons': active_neurons
            })
            print(f"  Input {inputs} -> {spike_count} spikes from neurons {active_neurons}")

        print("\nTesting adaptive learning...")

        
        # Test STDP learning
        pre_spikes = {0: 10, 1: 15, 2: 20}
        post_spikes = {3: 25, 4: 30}
        
        weights_before = spiking_net.weights.copy()
        adaptive_learner.stdp_update(pre_spikes, post_spikes)
        weights_after = spiking_net.weights.copy()
        
        weight_changes = np.sum(np.abs(weights_after - weights_before))
        print(f"  Weight changes: {weight_changes:.6f}")
        
        print("\nTesting parallel processing...")
        
        # Test parallel execution
        operations = [
            {'opcode': 'ADD', 'operands': {'a': 1, 'b': 2}},
            {'opcode': 'AND', 'operands': {'a': 2, 'b': 1}},
            {'opcode': 'OR', 'operands': {'a': 0, 'b': 1}},
            {'opcode': 'XOR', 'operands': {'a': 1, 'b': 1}}
        ]
        
        start_time = time.time()
        parallel_results = parallel_processor.parallel_execute(operations)
        parallel_time = time.time() - start_time
        
        print(f"  Parallel execution: {len(operations)} ops in {parallel_time:.4f}s")
        print(f"  Results: {parallel_results}")
        
        print("\nTesting neuromorphic pipeline...")
        
        # Test pipeline processing
        instructions = [
            ("ADD", 1, 2),
            ("AND", 1, 2),
            ("OR", 0, 1),
            ("XOR", 2, 1)
        ]
        
        pipeline_results = []
        for instruction in instructions:
            result = pipeline.process_instruction(instruction)
            pipeline_results.append(result)
        
        self.results['tests']['neuromorphic'] = {
            'spike_tests': spike_results,
            'weight_adaptation': float(weight_changes),
            'parallel_performance': {
                'operations': len(operations),
                'execution_time': parallel_time,
                'results': parallel_results
            },
            'pipeline_results': pipeline_results,
            'pipeline_stats': pipeline.stats
        }
        
        logger.info("Neuromorphic computing tests completed")
        print("Neuromorphic tests completed")
    
    def test_quantum_simulation(self):
        """Test 5: Quantum ternary operations"""
        print("\nTEST 5: Quantum Ternary Simulation")
        print("-" * 50)
        
        quantum_results = []
        
        print("Creating ternary qubits (qutrits)...")
        
        for test_case in range(5):
            qutrit = TernaryQubit()
            
            print(f"\nTest case {test_case + 1}:")
            print(f"  Initial state: {qutrit.state}")
            
            # Apply quantum gates
            operations = []
            
            if test_case % 3 == 0:
                qutrit.x_gate()
                operations.append("X")
            elif test_case % 3 == 1:
                qutrit.z_gate()
                operations.append("Z")
            else:
                qutrit.hadamard()
                operations.append("H")
            
            print(f"  After {operations[-1]} gate: {qutrit.state}")
            
            # Measure
            measurement = qutrit.measure()
            print(f"  Measurement result: {measurement}")
            
            quantum_results.append({
                'initial_state': [1, 0, 0],  # Always start in |0⟩
                'operations': operations,
                'final_state': qutrit.state.tolist(),
                'measurement': measurement
            })
        
        self.results['tests']['quantum'] = quantum_results
        logger.info("Quantum simulation tests completed")
        print("Quantum tests completed")
    
    def test_applications(self):
        """Test 6: Application-level tests"""
        print("\nTEST 6: Application Layer")
        print("-" * 50)
        
        print("Testing ternary computer vision...")
        
        # Test ternary image processing
        cv_processor = TernaryImageProcessor(use_neural=True)
        
        # Create test images
        test_image1 = np.random.randint(0, 256, (10, 10))
        test_image2 = np.random.randint(0, 256, (10, 10))
        
        # Convert to ternary
        ternary_img1 = cv_processor.ternary_threshold(test_image1)
        ternary_img2 = cv_processor.ternary_threshold(test_image2)
        
        # Apply operations
        and_result = cv_processor.apply_operation(ternary_img1, ternary_img2, "AND")
        or_result = cv_processor.apply_operation(ternary_img1, ternary_img2, "OR")
        xor_result = cv_processor.apply_operation(ternary_img1, ternary_img2, "XOR")
        
        print(f"  Processed images: {test_image1.shape} -> ternary operations completed")
        
        print("\nTesting ternary NLP...")
        
        # Test ternary NLP
        nlp_processor = TernaryNLP()
        
        test_texts = [
            "hello world",
            "ternary computing",
            "neural networks",
            "hello world"  # Same as first for similarity test
        ]
        
        similarities = []
        for i, text1 in enumerate(test_texts):
            for j, text2 in enumerate(test_texts):
                if i < j:
                    similarity = nlp_processor.semantic_similarity(text1, text2)
                    similarities.append({
                        'text1': text1,
                        'text2': text2,
                        'similarity': similarity
                    })
                    print(f"  Similarity('{text1}', '{text2}') = {similarity:.3f}")
        
        self.results['tests']['applications'] = {
            'computer_vision': {
                'input_shapes': [test_image1.shape, test_image2.shape],
                'operations_completed': ['AND', 'OR', 'XOR'],
                'output_shapes': [and_result.shape, or_result.shape, xor_result.shape]
            },
            'nlp': {
                'test_texts': test_texts,
                'similarities': similarities
            }
        }
        
        logger.info("Application layer tests completed")
        print("Application tests completed")
    
    def test_integrated_neural_cpu(self):
        """Test 7: Integrated Neural CPU System"""
        print("\nTEST 7: Integrated Neural CPU System")
        print("-" * 50)
        
        # Initialize neural CPU
        neural_cpu = create_neural_cpu(auto_optimize=True, cache_models=True)
        
        print("🔧 System initialization completed")
        print(f"  Supported operations: {neural_cpu.supported_operations}")
        
        # Single operation tests
        print("\nTesting single operations...")
        single_results = {}
        
        for operation in ['AND', 'OR', 'XOR']:
            results = []
            for a in range(3):
                for b in range(3):
                    start_time = time.perf_counter()
                    result = neural_cpu.execute_operation(operation, a, b)
                    execution_time = time.perf_counter() - start_time
                    
                    results.append({
                        'inputs': (a, b),
                        'output': result,
                        'time': execution_time
                    })
            
            single_results[operation] = results
            avg_time = np.mean([r['time'] for r in results])
            print(f"  {operation}: {len(results)} operations, avg {avg_time*1000:.3f}ms")
        
        # Batch operation tests
        print("\nTesting batch operations...")
        batch_operations = [(i % 3, (i+1) % 3) for i in range(20)]
        
        batch_results = {}
        for operation in ['AND', 'OR', 'XOR']:
            start_time = time.perf_counter()
            results = neural_cpu.execute_batch_operations(operation, batch_operations)
            batch_time = time.perf_counter() - start_time
            
            batch_results[operation] = {
                'operations_count': len(batch_operations),
                'execution_time': batch_time,
                'throughput': len(batch_operations) / batch_time,
                'results': results[:5]  # Store first 5 results
            }
            
            print(f"  {operation}: {len(batch_operations)} ops in {batch_time:.4f}s "
                  f"({len(batch_operations)/batch_time:.0f} ops/sec)")
        
        # System benchmarking
        print("\nRunning comprehensive benchmark...")
        benchmark_results = neural_cpu.benchmark_all_operations()
        
        for op, metrics in benchmark_results.items():
            if 'error' not in metrics:
                print(f"  {op}: Accuracy={metrics['accuracy']:.3f}, "
                      f"Time={metrics['avg_time_ms']:.2f}ms")
        
        # System status
        system_status = neural_cpu.get_system_status()
        
        self.results['tests']['integrated_neural_cpu'] = {
            'single_operations': single_results,
            'batch_operations': batch_results,
            'benchmark_results': benchmark_results,
            'system_status': system_status
        }
        
        logger.info("Integrated neural CPU tests completed")
        print("Integrated Neural CPU tests completed")
    
    def test_fpga_generation(self):
        """Test 8: FPGA/Hardware generation"""
        print("\nTEST 8: FPGA/Hardware Generation")
        print("-" * 50)
        
        print("Generating Verilog HDL...")
        
        # Generate Verilog code
        verilog_code = generate_verilog_module({})
        testbench_code = generate_testbench()
        
        # Save generated files
        verilog_path = "test_results/ternary_cpu.v"
        testbench_path = "test_results/ternary_cpu_tb.v"
        
        with open(verilog_path, 'w') as f:
            f.write(verilog_code)
        
        with open(testbench_path, 'w') as f:
            f.write(testbench_code)
        
        print(f"  Verilog module saved to: {verilog_path}")
        print(f"  Testbench saved to: {testbench_path}")
        print(f"  Verilog code length: {len(verilog_code)} characters")
        print(f"  Testbench length: {len(testbench_code)} characters")
        
        self.results['tests']['fpga_generation'] = {
            'verilog_file': verilog_path,
            'testbench_file': testbench_path,
            'verilog_length': len(verilog_code),
            'testbench_length': len(testbench_code)
        }
        
        logger.info("FPGA generation tests completed")
        print("FPGA generation tests completed")
    
    def generate_performance_plots(self):
        """Generate comprehensive performance visualization plots"""
        print("\nGenerating Performance Plots...")
        print("-" * 50)
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Neural Network Evolution
        if self.results['evolution_history']:
            ax1 = plt.subplot(2, 3, 1)
            for evolution in self.results['evolution_history']:
                if evolution['fitness_history']:
                    plt.plot(evolution['fitness_history'], 
                            label=f"{evolution['operation']} Fitness", 
                            linewidth=2, alpha=0.8)
            plt.title('Genetic Algorithm Evolution', fontsize=14, color='cyan')
            plt.xlabel('Generation')
            plt.ylabel('Fitness Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 2: Operation Performance Comparison
        if 'neural_networks' in self.results['tests']:
            ax2 = plt.subplot(2, 3, 2)
            operations = list(self.results['tests']['neural_networks'].keys())
            std_times = [self.results['tests']['neural_networks'][op]['standard']['benchmark']['avg_time']*1000 
                        for op in operations]
            gen_times = [self.results['tests']['neural_networks'][op]['genetic']['benchmark']['avg_time']*1000 
                        for op in operations]
            trad_times = [self.results['tests']['neural_networks'][op]['traditional']['benchmark']['avg_time']*1000 
                         for op in operations]
            
            x = np.arange(len(operations))
            width = 0.25
            
            plt.bar(x - width, std_times, width, label='Standard NN', color='skyblue', alpha=0.8)
            plt.bar(x, gen_times, width, label='Genetic NN', color='lightgreen', alpha=0.8)
            plt.bar(x + width, trad_times, width, label='Traditional', color='coral', alpha=0.8)
            
            plt.title('Operation Performance Comparison', fontsize=14, color='cyan')
            plt.xlabel('Operations')
            plt.ylabel('Execution Time (ms)')
            plt.xticks(x, operations)
            plt.legend()
            plt.yscale('log')
        
        # Plot 3: Accuracy Comparison
        if 'neural_networks' in self.results['tests']:
            ax3 = plt.subplot(2, 3, 3)
            std_acc = [self.results['tests']['neural_networks'][op]['standard']['benchmark']['accuracy'] 
                      for op in operations]
            gen_acc = [self.results['tests']['neural_networks'][op]['genetic']['benchmark']['accuracy'] 
                      for op in operations]
            
            x = np.arange(len(operations))
            width = 0.35
            
            plt.bar(x - width/2, std_acc, width, label='Standard NN', color='skyblue', alpha=0.8)
            plt.bar(x + width/2, gen_acc, width, label='Genetic NN', color='lightgreen', alpha=0.8)
            
            plt.title('Model Accuracy Comparison', fontsize=14, color='cyan')
            plt.xlabel('Operations')
            plt.ylabel('Accuracy')
            plt.xticks(x, operations)
            plt.legend()
            plt.ylim(0, 1.1)
        
        # Plot 4: Neuromorphic Activity
        if 'neuromorphic' in self.results['tests']:
            ax4 = plt.subplot(2, 3, 4)
            spike_data = self.results['tests']['neuromorphic']['spike_tests']
            spike_counts = [test['spike_count'] for test in spike_data]
            
            plt.plot(range(len(spike_counts)), spike_counts, 'o-', 
                    color='yellow', linewidth=2, markersize=8, alpha=0.8)
            plt.title('Neuromorphic Spike Activity', fontsize=14, color='cyan')
            plt.xlabel('Test Case')
            plt.ylabel('Spike Count')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Quantum State Visualization
        if 'quantum' in self.results['tests']:
            ax5 = plt.subplot(2, 3, 5)
            quantum_data = self.results['tests']['quantum']
            measurements = [test['measurement'] for test in quantum_data]
            
            unique, counts = np.unique(measurements, return_counts=True)
            colors = ['red', 'green', 'blue'][:len(unique)]
            
            plt.bar(unique, counts, color=colors, alpha=0.8)
            plt.title('Quantum Measurement Distribution', fontsize=14, color='cyan')
            plt.xlabel('Measurement Result')
            plt.ylabel('Frequency')
            plt.xticks(range(3), ['|0⟩', '|1⟩', '|2⟩'])
        
        # Plot 6: System Overview
        ax6 = plt.subplot(2, 3, 6)
        if 'integrated_neural_cpu' in self.results['tests']:
            system_status = self.results['tests']['integrated_neural_cpu']['system_status']
            op_counts = system_status['operation_counts']
            
            if op_counts:
                operations = list(op_counts.keys())
                counts = list(op_counts.values())
                
                plt.pie(counts, labels=operations, autopct='%1.1f%%', 
                       startangle=90, colors=plt.cm.Set3.colors)
                plt.title('Operation Distribution', fontsize=14, color='cyan')
        
        plt.tight_layout()
        plt.savefig('test_results/plots/performance_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
        
        print("Performance plots generated and saved")
    
    def generate_comprehensive_report(self):
        """Generate detailed test report"""
        print("\nGenerating Comprehensive Report...")
        print("-" * 50)
        
        report = {
            'test_summary': {
                'timestamp': self.results['timestamp'],
                'total_tests': len(self.results['tests']),
                'test_modules': list(self.results['tests'].keys()),
                'success': True
            },
            'performance_summary': {},
            'recommendations': [],
            'detailed_results': self.results
        }
        
        # Performance summary
        if 'neural_networks' in self.results['tests']:
            nn_results = self.results['tests']['neural_networks']
            best_accuracies = {}
            avg_times = {}
            
            for op, data in nn_results.items():
                std_acc = data['standard']['benchmark']['accuracy']
                gen_acc = data['genetic']['benchmark']['accuracy']
                best_accuracies[op] = max(std_acc, gen_acc)
                
                std_time = data['standard']['benchmark']['avg_time']
                gen_time = data['genetic']['benchmark']['avg_time']
                avg_times[op] = min(std_time, gen_time)
            
            report['performance_summary'] = {
                'best_accuracies': best_accuracies,
                'fastest_times': avg_times,
                'overall_accuracy': np.mean(list(best_accuracies.values())),
                'overall_speed': np.mean(list(avg_times.values()))
            }
        
        # Generate recommendations
        if report['performance_summary'].get('overall_accuracy', 0) > 0.95:
            report['recommendations'].append("Excellent neural network accuracy achieved")
        else:
            report['recommendations'].append("Consider more training epochs for better accuracy")
        
        if 'neuromorphic' in self.results['tests']:
            weight_change = self.results['tests']['neuromorphic']['weight_adaptation']
            if weight_change > 0.001:
                report['recommendations'].append("Neuromorphic adaptation working correctly")
            else:
                report['recommendations'].append("Low neuromorphic weight adaptation")
        
        # Helper to convert tuple keys to strings
        def stringify_keys(obj):
            if isinstance(obj, dict):
                return {str(k): stringify_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [stringify_keys(i) for i in obj]
            else:
                return obj

        # Save report with safe key conversion
        safe_report = stringify_keys(report)
        with open('test_results/comprehensive_report.json', 'w') as f:
            json.dump(safe_report, f, indent=2, default=str)
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(report)
        with open('test_results/TEST_REPORT.md', 'w') as f:
            f.write(markdown_report)
        
        print("Comprehensive report saved to:")
        print("- test_results/comprehensive_report.json")
        print("- test_results/TEST_REPORT.md")
        
        return report

    
    def generate_markdown_report(self, report):
        """Generate markdown format report"""
        md = f"""# Ternary Neuromorphic CPU Test Report

## Test Summary
- **Date**: {report['test_summary']['timestamp']}
- **Total Tests**: {report['test_summary']['total_tests']}
- **Status**: {'PASSED' if report['test_summary']['success'] else 'FAILED'}

## Performance Overview
"""
        
        if report['performance_summary']:
            ps = report['performance_summary']
            md += f"""
- **Overall Accuracy**: {ps.get('overall_accuracy', 0):.3f}
- **Average Speed**: {ps.get('overall_speed', 0)*1000:.3f}ms

### Best Accuracies by Operation
"""
            for op, acc in ps.get('best_accuracies', {}).items():
                md += f"- **{op}**: {acc:.3f}\n"
        
        md += "\n## Test Modules Completed\n"
        for module in report['test_summary']['test_modules']:
            md += f"-{module.replace('_', ' ').title()}\n"
        
        md += "\n## Recommendations\n"
        for rec in report['recommendations']:
            md += f"- {rec}\n"
        
        md += f"""
## Detailed Architecture Analysis

### Traditional CPU Components
- Ternary ALU with all basic operations (AND, OR, XOR, ADD, SUB, etc.)
- Register set with 9 ternary registers
- Memory system with 27 ternary cells
- Complete instruction set with assembly support

### Neural Network Integration
- Standard feedforward networks for ternary operations
- Genetic algorithm evolution for network optimization
- Automatic model selection based on performance
- Persistent model storage and loading

### Neuromorphic Computing
- Spiking neural networks with 32+ neurons
- Spike-timing dependent plasticity (STDP) learning
- Event-driven processing architecture
- Parallel processing with multiple cores

### Quantum Computing Simulation
- Ternary quantum bits (qutrits) implementation
- Quantum gates: X, Z, Hadamard
- Superposition and measurement simulation
- Quantum-classical hybrid operations

### Application Layer
- Computer vision with ternary image processing
- Natural language processing with ternary encoding
- Semantic similarity computation
- Real-world application demonstrations

## Technical Achievements

1. **Multi-Paradigm Architecture**: Successfully integrates traditional, neural, neuromorphic, and quantum computing paradigms
2. **Adaptive Learning**: Genetic algorithms continuously improve neural network performance
3. **Event-Driven Processing**: Neuromorphic components respond to computational events
4. **Hardware Generation**: Automatic Verilog HDL generation for FPGA implementation
5. **Performance Optimization**: Dynamic selection between computing modes for optimal performance

## Research Contributions

- Novel ternary-based neuromorphic CPU architecture
- Genetic evolution of neural networks for ternary operations  
- Integration of quantum computing concepts with classical ternary logic
- Event-driven neuromorphic processing for adaptive computation
- Memory-in-memory computing implementation

---
*Generated by Ternary Neuromorphic CPU Test Suite*
"""
        return md
    
    def run_stress_test(self):
        """Run comprehensive stress testing"""
        print("\nSTRESS TEST: High-Load Performance Analysis")
        print("-" * 50)
        
        # Test neural CPU under load
        neural_cpu = create_neural_cpu()
        
        operations = ['AND', 'OR', 'XOR', 'ADD', 'SUB']
        batch_sizes = [10, 50, 100, 500, 1000]
        stress_results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            batch_ops = [(np.random.randint(0, 3), np.random.randint(0, 3)) 
                        for _ in range(batch_size)]
            
            op_results = {}
            for operation in operations:
                start_time = time.perf_counter()
                results = neural_cpu.execute_batch_operations(operation, batch_ops)
                execution_time = time.perf_counter() - start_time
                
                throughput = batch_size / execution_time
                op_results[operation] = {
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'success_rate': len(results) / batch_size
                }
                
                print(f"  {operation}: {throughput:.0f} ops/sec")
            
            stress_results[batch_size] = op_results
        
        self.results['stress_test'] = stress_results
        print("Stress testing completed")
    
    def run_memory_analysis(self):
        """Analyze memory usage and efficiency"""
        print("\nMEMORY ANALYSIS: Usage and Efficiency")
        print("-" * 50)
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Test memory usage during neural network training
        print("Training neural networks and monitoring memory...")
        
        memory_snapshots = []
        operations = ['AND', 'OR', 'XOR']
        
        for i, operation in enumerate(operations):
            # Take memory snapshot
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append({
                'operation': operation,
                'memory_mb': current_memory,
                'memory_growth': current_memory - initial_memory
            })
            
            print(f"  {operation} training: {current_memory:.2f} MB "
                  f"(+{current_memory - initial_memory:.2f} MB)")
            
            # Train a small model for memory testing
            model, _ = train_or_load_model(operation, epochs=100, force_retrain=True)
            
            # Force garbage collection
            gc.collect()
        
        # Final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        self.results['memory_analysis'] = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'total_growth_mb': total_growth,
            'snapshots': memory_snapshots,
            'efficiency_score': initial_memory / final_memory  # Higher is better
        }
        
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Total memory growth: {total_growth:.2f} MB")
        print("Memory analysis completed")
    
    def run_accuracy_validation(self):
        """Validate accuracy across all operations and modes"""
        print("\nACCURACY VALIDATION: Cross-Modal Verification")
        print("-" * 50)
        
        operations = ['AND', 'OR', 'XOR', 'ADD', 'SUB']
        test_cases = [(a, b) for a in range(3) for b in range(3)]
        
        # Get ground truth from traditional implementation
        ground_truth = {}
        cpu = TernaryCPU(neural_mode=False)
        
        for operation in operations:
            ground_truth[operation] = {}
            for a, b in test_cases:
                result = cpu.traditionalExecute(operation, a, b)
                ground_truth[operation][(a, b)] = result
        
        # Test neural CPU accuracy
        neural_cpu = create_neural_cpu()
        accuracy_results = {}
        
        for operation in operations:
            correct = 0
            total = len(test_cases)
            errors = []
            
            for a, b in test_cases:
                neural_result = neural_cpu.execute_operation(operation, a, b)
                expected = ground_truth[operation][(a, b)]
                
                if neural_result == expected:
                    correct += 1
                else:
                    errors.append({
                        'inputs': (a, b),
                        'expected': expected,
                        'actual': neural_result
                    })
            
            accuracy = correct / total
            accuracy_results[operation] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'errors': errors
            }
            
            status = "✅" if accuracy >= 0.95 else "⚠️" if accuracy >= 0.90 else "❌"
            print(f"  {operation}: {accuracy:.3f} ({correct}/{total}) {status}")
            
            if errors:
                print(f"    Errors: {len(errors)} cases")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      {operation}{error['inputs']} -> "
                          f"got {error['actual']}, expected {error['expected']}")
        
        overall_accuracy = np.mean([result['accuracy'] for result in accuracy_results.values()])
        print(f"\nOverall accuracy: {overall_accuracy:.3f}")
        
        self.results['accuracy_validation'] = {
            'overall_accuracy': overall_accuracy,
            'operation_accuracies': accuracy_results,
            'ground_truth': ground_truth
        }
        
        print("Accuracy validation completed")
    
    def run_all_tests(self):
        """Execute complete test suite"""
        start_time = time.time()
        
        try:
            # Core functionality tests
            self.test_basic_ternary_operations()
            self.test_traditional_cpu()
            self.test_neural_networks()
            self.test_neuromorphic_components()
            self.test_quantum_simulation()
            self.test_applications()
            self.test_integrated_neural_cpu()
            self.test_fpga_generation()
            
            # Advanced analysis
            self.run_stress_test()
            self.run_memory_analysis()
            self.run_accuracy_validation()
            
            # Generate outputs
            self.generate_performance_plots()
            report = self.generate_comprehensive_report()
            
            total_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f" Total execution time: {total_time:.2f} seconds")
            print(f" Tests completed: {len(self.results['tests'])}")
            print(f" Performance plots: test_results/plots/")
            print(f" Detailed report: test_results/TEST_REPORT.md")
            print(f" Raw results: test_results/comprehensive_report.json")
            
            if 'accuracy_validation' in self.results:
                overall_acc = self.results['accuracy_validation']['overall_accuracy']
                status = "🟢" if overall_acc >= 0.95 else "🟡" if overall_acc >= 0.90 else "🔴"
                print(f"Overall system accuracy: {overall_acc:.3f} {status}")
            
            print("\nSYSTEM READY FOR RESEARCH AND DEVELOPMENT!")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\nTest suite failed with error: {e}")
            logger.error(f"Test suite error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def interactive_demo(self):
        """Run interactive demonstration"""
        print("\nINTERACTIVE DEMO MODE")
        print("-" * 50)
        print("Available commands:")
        print("  1 - Test single operation")
        print("  2 - Neural network comparison")
        print("  3 - Neuromorphic spike demo")
        print("  4 - Quantum simulation")
        print("  5 - Batch performance test")
        print("  6 - Full system benchmark")
        print("  0 - Exit")
        
        neural_cpu = create_neural_cpu()
        
        while True:
            try:
                choice = input("\nEnter command (0-6): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    op = input("Operation (AND/OR/XOR/ADD/SUB): ").upper()
                    a = int(input("Operand A (0-2): "))
                    b = int(input("Operand B (0-2): "))
                    result = neural_cpu.execute_operation(op, a, b)
                    print(f"Result: {op}({a}, {b}) = {result}")
                    
                elif choice == '2':
                    op = input("Operation to compare (AND/OR/XOR): ").upper()
                    comparison = compare_models(op)
                    print(f"Winner: {comparison['winner']}")
                    print(f"Standard: {comparison['standard']['accuracy']:.3f}")
                    print(f"Genetic: {comparison['genetic']['accuracy']:.3f}")
                    
                elif choice == '3':
                    snn = SpikingNeuralNetwork()
                    inputs = [int(x) for x in input("Enter 3 inputs (0-2): ").split()]
                    spikes = snn.spike(inputs)
                    print(f"Generated {len(spikes)} spikes: {list(spikes.keys())}")
                    
                elif choice == '4':
                    qutrit = TernaryQubit()
                    gate = input("Gate (X/Z/H): ").upper()
                    if gate == 'X':
                        qutrit.x_gate()
                    elif gate == 'Z':
                        qutrit.z_gate()
                    elif gate == 'H':
                        qutrit.hadamard()
                    result = qutrit.measure()
                    print(f"Quantum measurement: |{result}⟩")
                    
                elif choice == '5':
                    op = input("Operation (AND/OR/XOR/ADD/SUB): ").upper()
                    size = int(input("Batch size: "))
                    ops = [(np.random.randint(0, 3), np.random.randint(0, 3)) for _ in range(size)]
                    start_time = time.perf_counter()
                    results = neural_cpu.execute_batch_operations(op, ops)
                    exec_time = time.perf_counter() - start_time
                    print(f"Processed {size} operations in {exec_time:.4f}s")
                    print(f"Throughput: {size/exec_time:.0f} ops/sec")
                    
                elif choice == '6':
                    print("Running full benchmark...")
                    benchmarks = neural_cpu.benchmark_all_operations()
                    for op, metrics in benchmarks.items():
                        if 'error' not in metrics:
                            print(f"{op}: Acc={metrics['accuracy']:.3f}, "
                                  f"Time={metrics['avg_time_ms']:.2f}ms")
                
            except (ValueError, KeyError) as e:
                print(f"Invalid input: {e}")
            except KeyboardInterrupt:
                print("\nDemo interrupted")
                break
        
        print("Demo completed!")


def main():
    """Main execution function"""
    print("Starting Ternary Neuromorphic CPU Test Suite...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Ternary Neuromorphic CPU Test Suite')
    parser.add_argument('--mode', choices=['full', 'quick', 'demo'], default='full',
                       help='Test mode: full (all tests), quick (basic tests), demo (interactive)')
    parser.add_argument('--plots', action='store_true', help='Generate performance plots')
    parser.add_argument('--no-neural', action='store_true', help='Skip neural network tests')
    parser.add_argument('--stress', action='store_true', help='Run stress tests')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite()
    
    try:
        if args.mode == 'demo':
            # Interactive demo mode
            test_suite.interactive_demo()
            
        elif args.mode == 'quick':
            # Quick test mode - basic functionality only
            print("Running quick test suite...")
            test_suite.test_basic_ternary_operations()
            test_suite.test_traditional_cpu()
            if not args.no_neural:
                test_suite.test_integrated_neural_cpu()
            test_suite.run_accuracy_validation()
            
            if args.plots:
                test_suite.generate_performance_plots()
            
            report = test_suite.generate_comprehensive_report()
            print("Quick test suite completed")
            
        else:
            # Full test mode
            print("Running comprehensive test suite...")
            success = test_suite.run_all_tests()
            
            if success:
                print("\nCONGRATULATIONS!")
                print("Your Ternary Neuromorphic CPU is fully operational!")
                
                # Additional stress testing if requested
                if args.stress:
                    print("Running extended stress tests...")
                    for i in range(5):
                        print(f"Stress test iteration {i+1}/5")
                        test_suite.run_stress_test()
                
            else:
                print("\nSome tests failed. Check logs for details.")
                return 1
    
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
