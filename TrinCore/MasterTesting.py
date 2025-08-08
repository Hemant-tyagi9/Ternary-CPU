#!/usr/bin/env python3

import sys
import unittest
import numpy as np
import json
import time
from typing import Dict, List, Any
import traceback
import warnings
import argparse

from trincore_applications.config import TrinCoreConfig
from trincore_applications.logging import logger

class TrinCoreTestRunner:
    def __init__(self):
        # Initialize configuration
        self.config = TrinCoreConfig()
        self.config.set("system.log_level", "WARNING")  # Reduce log noise during tests
        
        self.test_results = {
            'passed': [],
            'failed': [],
            'errors': [],
            'performance': {},
            'config': self.config._config
        }
        self.start_time = time.time()
        
    def log_result(self, test_name: str, status: str, error=None, perf_data=None):
        """Enhanced test logging"""
        result = {
            'test': test_name,
            'status': status,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': self.config._config
        }
        
        if error:
            result['error'] = str(error)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Test failed: {test_name} - {str(error)}")
            self.test_results['errors'].append(result)
        elif status == 'passed':
            self.test_results['passed'].append(result)
            if perf_data:
                self.test_results['performance'][test_name] = perf_data
            logger.info(f"Test passed: {test_name}")
        else:
            logger.warning(f"Test failed: {test_name}")
            self.test_results['failed'].append(result)
            
    def run_tests(self):
        """Execute all test suites with enhanced logging"""
        logger.info("🚀 Starting TrinCore Comprehensive Test Suite")
        
        # Core component tests
        self.test_cpu_components()
        self.test_alu_operations()
        self.test_nn_training() 
        self.test_nn_integration()
        
        # Advanced feature tests
        self.test_quantum_simulation()
        self.test_applications()
        self.test_full_pipeline()
        
        # Neuromorphic tests
        if self.config.get("neuromorphic.enabled", True):
            logger.info("\n🧠 Testing Neuromorphic Features...")
            self.test_neuromorphic_features()
            self.test_spiking_neural_networks()
            self.test_event_driven_processing()
            self.test_in_memory_computing()
            self.test_real_time_adaptation()
            self.test_neuromorphic_performance_comparison()
        
        self.generate_report()
    
    def test_cpu_components(self):
        """Test core CPU components with config support"""
        test_name = "CPU Core Components"
        try:
            from CPU_Components.ternary_memory import TernaryMemory
            from CPU_Components.register_sets import RegisterSet
            from CPU_Components.program_counter import ProgramCounter
            
            mem_size = self.config.get("memory.test_size", 27)
            mem = TernaryMemory(size=mem_size)
            mem.store(0, 2)
            assert mem.load(0) == 2, "Memory read/write failed"
            
            reg_count = self.config.get("registers.test_count", 9)
            reg = RegisterSet(num_registers=reg_count)
            reg.write(0, 1)
            assert reg.read(0) == 1, "Register read/write failed"
            
            pc = ProgramCounter()
            pc.increment()
            assert pc.get() == 1, "PC increment failed"
            
            self.log_result(test_name, 'passed')
        except Exception as e:
            self.log_result(test_name, 'failed', e)
    
    def test_alu_operations(self):
        """Test ALU operations with config support"""
        test_name = "ALU Operations"
        try:
            from CPU_Components.alu import TernaryALU
            from NN.nn_integration import NeuralIntegration
            
            # Traditional ALU
            alu = TernaryALU()
            assert alu.execute("ADD", 1, 2) == 0, "Traditional ADD failed"
            
            # Neural ALU with config
            neural_alu = NeuralIntegration()
            ops = self.config.get("nn.test_operations", ["ADD"])
            neural_alu.train_models(ops)
            
            result = neural_alu.execute_operation("ADD", 1, 2)
            assert result in [0, 1, 2], "Neural ADD returned invalid ternary value"
            
            # Test batch operations
            if self.config.get("nn.test_batch_ops", True):
                batch_results = neural_alu.execute_batch_operations(
                    "ADD", [(1,2), (2,2), (0,1)]
                )
                assert all(r in [0,1,2] for r in batch_results), "Batch ops failed"
            
            self.log_result(test_name, 'passed')
        except Exception as e:
            self.log_result(test_name, 'failed', e)
    
    # ... (keep other test methods similar but add logger where appropriate)
    
    def generate_report(self):
        """Enhanced report generation with config info"""
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        elapsed_time = time.time() - self.start_time
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 TrinCore Test Report")
        logger.info(f"⏱️  Duration: {elapsed_time:.2f}s")
        logger.info(f"✅ Passed: {len(self.test_results['passed'])}/{total_tests}")
        logger.info(f"❌ Failed: {len(self.test_results['failed'])}/{total_tests}")
        logger.info(f"🔥 Errors: {len(self.test_results['errors'])}")
        logger.info(f"{'='*50}\n")
        
        # Save detailed report with config
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': len(self.test_results['passed']),
                'failed': len(self.test_results['failed']),
                'errors': len(self.test_results['errors']),
                'duration_seconds': elapsed_time,
                'config': self.test_results['config']
            },
            'details': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open('trincore_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📊 Full report saved to 'trincore_test_report.json'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrinCore Master Test Suite')
    parser.add_argument('--verbose', action='store_true', help='Show detailed error output')
    parser.add_argument('--config', help='Path to custom config file')
    args = parser.parse_args()
    
    if args.verbose:
        TrinCoreConfig().set("system.log_level", "DEBUG")
    
    if args.config:
        TrinCoreConfig().load(args.config)
    
    tester = TrinCoreTestRunner()
    tester.run_tests()
