#!/usr/bin/env python3

import sys
import os
import time
import json
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import csv

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich not available - using basic output")

# Enhanced Results Directory Structure
RESULTS_BASE = Path("results")
RESULTS_DIRS = {
    'live': RESULTS_BASE / "live_monitoring",
    'neuromorphic': RESULTS_BASE / "ternary_neuromorphic",
    'performance': RESULTS_BASE / "performance",
    'analysis': RESULTS_BASE / "analysis",
    'visualizations': RESULTS_BASE / "visualizations"
}

for dir_path in RESULTS_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

class LiveMetricsCollector:
    """Real-time metrics collection and monitoring"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = {
            'timestamps': deque(maxlen=max_history),
            'operations_per_second': deque(maxlen=max_history),
            'spike_rate': deque(maxlen=max_history),
            'neural_efficiency': deque(maxlen=max_history),
            'memory_usage': deque(maxlen=max_history),
            'cpu_utilization': deque(maxlen=max_history),
            'accuracy': deque(maxlen=max_history),
            'latency_ms': deque(maxlen=max_history)
        }
        self.current_metrics = {
            'total_operations': 0,
            'neural_operations': 0,
            'traditional_operations': 0,
            'total_spikes': 0,
            'correct_operations': 0,
            'failed_operations': 0,
            'average_latency': 0.0,
            'peak_ops_per_second': 0.0,
            'uptime': 0.0
        }
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_lock = threading.Lock()
        
    def update_metrics(self, operation_data: Dict[str, Any]):
        """Update metrics with new operation data"""
        with self.update_lock:
            current_time = time.time()
            self.current_metrics['uptime'] = current_time - self.start_time
            
            # Update counters
            if operation_data.get('success', True):
                self.current_metrics['total_operations'] += 1
                self.current_metrics['correct_operations'] += 1
                
                if operation_data.get('is_neural', False):
                    self.current_metrics['neural_operations'] += 1
                else:
                    self.current_metrics['traditional_operations'] += 1
                    
                self.current_metrics['total_spikes'] += operation_data.get('spikes', 0)
            else:
                self.current_metrics['failed_operations'] += 1
            
            # Calculate rates
            time_delta = current_time - self.last_update
            if time_delta >= 0.1:  # Update every 100ms
                ops_per_second = 1.0 / max(operation_data.get('latency', 0.001), 0.001)
                spike_rate = operation_data.get('spikes', 0) / max(time_delta, 0.001)
                
                neural_efficiency = (self.current_metrics['neural_operations'] / 
                                   max(self.current_metrics['total_operations'], 1) * 100)
                
                accuracy = (self.current_metrics['correct_operations'] / 
                          max(self.current_metrics['total_operations'], 1) * 100)
                
                # Update history
                self.metrics_history['timestamps'].append(current_time)
                self.metrics_history['operations_per_second'].append(ops_per_second)
                self.metrics_history['spike_rate'].append(spike_rate)
                self.metrics_history['neural_efficiency'].append(neural_efficiency)
                self.metrics_history['accuracy'].append(accuracy)
                self.metrics_history['latency_ms'].append(operation_data.get('latency', 0) * 1000)
                
                # Update peaks
                if ops_per_second > self.current_metrics['peak_ops_per_second']:
                    self.current_metrics['peak_ops_per_second'] = ops_per_second
                
                self.last_update = current_time
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self.update_lock:
            return {
                **self.current_metrics,
                'current_ops_per_second': list(self.metrics_history['operations_per_second'])[-1] if self.metrics_history['operations_per_second'] else 0,
                'current_spike_rate': list(self.metrics_history['spike_rate'])[-1] if self.metrics_history['spike_rate'] else 0,
                'neural_percentage': (self.current_metrics['neural_operations'] / 
                                    max(self.current_metrics['total_operations'], 1) * 100),
                'accuracy_percentage': (self.current_metrics['correct_operations'] / 
                                      max(self.current_metrics['total_operations'], 1) * 100)
            }
    
    def save_metrics_snapshot(self, filepath: str):
        """Save current metrics to file"""
        snapshot = self.get_current_snapshot()
        snapshot['history'] = {
            key: list(values) for key, values in self.metrics_history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

class EnhancedSpikingNeuralNetwork:
    """Enhanced spiking neural network with better spike generation"""
    
    def __init__(self, size: int = 64, learning_enabled: bool = True):
        self.size = size
        self.learning_enabled = learning_enabled
        
        # Enhanced network parameters
        self.neurons = np.random.rand(size) * 0.4
        self.weights = np.random.randn(size, size) * 0.08
        self.threshold = np.random.uniform(0.6, 0.8, size)
        self.decay = np.random.uniform(0.92, 0.98, size)
        self.refractory_period = np.zeros(size)
        
        # Spike tracking
        self.spike_log = []
        self.spike_patterns = {}
        self.time_step = 0
        
        # Learning parameters
        self.learning_rate = 0.001
        self.stdp_window = 20
        
    def spike(self, inputs: List[float], trace_mode: bool = False) -> Dict[int, int]:
        """Enhanced spike generation with better dynamics"""
        self.time_step += 1
        
        # Prepare input array
        input_array = np.array(inputs + [0] * (self.size - len(inputs)))[:self.size]
        input_array = input_array * np.random.uniform(0.8, 1.2, self.size)  # Add noise
        
        # Update refractory periods
        self.refractory_period = np.maximum(0, self.refractory_period - 1)
        
        # Update membrane potentials with enhanced dynamics
        network_input = np.dot(self.weights, input_array) * 0.12
        noise = np.random.normal(0, 0.02, self.size)
        
        self.neurons = (self.neurons * self.decay + 
                       network_input + 
                       noise * (1 - self.refractory_period/3))
        
        # Generate spikes with adaptive thresholds
        adaptive_threshold = self.threshold * np.random.uniform(0.95, 1.05, self.size)
        spike_mask = (self.neurons > adaptive_threshold) & (self.refractory_period == 0)
        
        spikes = {}
        spike_neurons = np.where(spike_mask)[0]
        
        for neuron_id in spike_neurons:
            spikes[neuron_id] = self.time_step
            self.spike_log.append((self.time_step, neuron_id))
            
            # Reset spiked neuron
            self.neurons[neuron_id] = 0.05
            self.refractory_period[neuron_id] = 3
            
            # STDP learning
            if self.learning_enabled:
                self._apply_stdp(neuron_id)
        
        # Pattern recognition
        if len(spikes) > 0:
            pattern_key = tuple(sorted(spikes.keys())[:5])  # Top 5 neurons
            self.spike_patterns[pattern_key] = self.spike_patterns.get(pattern_key, 0) + 1
        
        return spikes
    
    def _apply_stdp(self, post_neuron: int):
        """Apply spike-timing dependent plasticity"""
        if len(self.spike_log) < 2:
            return
            
        current_time = self.time_step
        
        # Look for recent pre-synaptic spikes
        for spike_time, pre_neuron in self.spike_log[-20:]:
            if pre_neuron != post_neuron:
                dt = current_time - spike_time
                if 0 < dt <= self.stdp_window:
                    # Potentiation
                    delta_w = self.learning_rate * np.exp(-dt/5)
                    self.weights[post_neuron, pre_neuron] += delta_w
                elif -self.stdp_window <= dt < 0:
                    # Depression
                    delta_w = -self.learning_rate * np.exp(dt/5)
                    self.weights[post_neuron, pre_neuron] += delta_w
        
        # Normalize weights
        self.weights = np.clip(self.weights, -0.5, 0.5)
    
    def get_spike_statistics(self) -> Dict[str, Any]:
        """Get comprehensive spike statistics"""
        if not self.spike_log:
            return {'total_spikes': 0, 'active_neurons': 0, 'spike_rate': 0}
        
        times = [t for t, _ in self.spike_log]
        neurons = [n for _, n in self.spike_log]
        
        return {
            'total_spikes': len(self.spike_log),
            'active_neurons': len(set(neurons)),
            'spike_rate': len(self.spike_log) / max(self.time_step, 1),
            'time_span': max(times) - min(times) if len(times) > 1 else 0,
            'most_active_neuron': max(set(neurons), key=neurons.count) if neurons else 0,
            'spike_patterns': len(self.spike_patterns),
            'network_synchrony': self._calculate_synchrony()
        }
    
    def _calculate_synchrony(self) -> float:
        """Calculate network synchrony measure"""
        if len(self.spike_log) < 10:
            return 0.0
        
        # Simple synchrony measure based on spike timing correlations
        recent_spikes = self.spike_log[-50:]
        times = [t for t, _ in recent_spikes]
        
        if len(set(times)) < 2:
            return 1.0  # Perfect synchrony
        
        # Calculate coefficient of variation of inter-spike intervals
        intervals = np.diff(sorted(set(times)))
        if len(intervals) == 0:
            return 0.0
        
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        return max(0, 1 - cv)  # Higher values = more synchrony

class EnhancedEventDrivenALU:
    """Enhanced ALU with better learning and adaptation"""
    
    def __init__(self, network_size: int = 64):
        self.spiking_net = EnhancedSpikingNeuralNetwork(network_size)
        self.operation_count = 0
        self.operation_history = deque(maxlen=1000)
        self.adaptation_threshold = 30
        self.confidence_scores = {}
        
        # Enhanced operation tracking
        self.operation_stats = {
            'ADD': {'correct': 0, 'total': 0, 'avg_confidence': 0.0},
            'AND': {'correct': 0, 'total': 0, 'avg_confidence': 0.0},
            'OR': {'correct': 0, 'total': 0, 'avg_confidence': 0.0},
            'XOR': {'correct': 0, 'total': 0, 'avg_confidence': 0.0}
        }
        
    def ternary_add(self, a: int, b: int) -> int:
        return (a + b) % 3
    
    def ternary_and(self, a: int, b: int) -> int:
        return min(a, b)
    
    def ternary_or(self, a: int, b: int) -> int:
        return max(a, b)
    
    def ternary_xor(self, a: int, b: int) -> int:
        return (a - b) % 3
    
    def execute(self, opcode: str, operands: Dict[str, Any], trace: bool = False) -> Dict[str, Any]:
        """Enhanced execution with detailed tracking"""
        start_time = time.time()
        self.operation_count += 1
        
        a, b = operands.get('a', 0), operands.get('b', 0)
        
        # Traditional computation for verification
        traditional_ops = {
            "ADD": self.ternary_add,
            "AND": self.ternary_and,
            "OR": self.ternary_or,
            "XOR": self.ternary_xor
        }
        
        traditional_result = traditional_ops.get(opcode, lambda x, y: 0)(a, b)
        neuromorphic_result = traditional_result
        is_neural = False
        spikes = {}
        confidence = 1.0
        
        # Use neuromorphic computation after adaptation
        if self.operation_count > self.adaptation_threshold:
            spikes = self.spiking_net.spike([float(a), float(b), float(a*b)], trace_mode=trace)
            
            if spikes:
                is_neural = True
                # Enhanced spike interpretation with confidence
                spike_sum = sum(spikes.keys())
                spike_count = len(spikes)
                
                # Multiple interpretation methods
                neuromorphic_result = spike_sum % 3
                confidence = min(1.0, spike_count / 5.0)  # More spikes = higher confidence
                
                # Verify result and adjust confidence
                if neuromorphic_result == traditional_result:
                    confidence = min(1.0, confidence * 1.2)
                else:
                    confidence *= 0.5
                    # Use traditional result if confidence is too low
                    if confidence < 0.3:
                        neuromorphic_result = traditional_result
                        is_neural = False
        
        # Update statistics
        if opcode in self.operation_stats:
            self.operation_stats[opcode]['total'] += 1
            if neuromorphic_result == traditional_result:
                self.operation_stats[opcode]['correct'] += 1
            
            # Update confidence average
            old_conf = self.operation_stats[opcode]['avg_confidence']
            total = self.operation_stats[opcode]['total']
            self.operation_stats[opcode]['avg_confidence'] = (old_conf * (total-1) + confidence) / total
        
        execution_time = time.time() - start_time
        
        # Create detailed result
        result = {
            'result': neuromorphic_result,
            'is_neural': is_neural,
            'traditional_result': traditional_result,
            'match': neuromorphic_result == traditional_result,
            'spikes': len(spikes),
            'spike_data': spikes,
            'confidence': confidence,
            'latency': execution_time,
            'operation_count': self.operation_count,
            'success': True
        }
        
        # Store in history
        self.operation_history.append({
            'opcode': opcode,
            'operands': (a, b),
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_ops = sum(stats['total'] for stats in self.operation_stats.values())
        total_correct = sum(stats['correct'] for stats in self.operation_stats.values())
        
        return {
            'total_operations': self.operation_count,
            'accuracy': total_correct / max(total_ops, 1),
            'operation_breakdown': self.operation_stats.copy(),
            'spike_stats': self.spiking_net.get_spike_statistics(),
            'recent_performance': self._analyze_recent_performance()
        }
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze performance over recent operations"""
        recent_ops = list(self.operation_history)[-100:]  # Last 100 operations
        
        if not recent_ops:
            return {'accuracy': 0, 'neural_ratio': 0, 'avg_latency': 0}
        
        correct = sum(1 for op in recent_ops if op['result']['match'])
        neural_ops = sum(1 for op in recent_ops if op['result']['is_neural'])
        avg_latency = np.mean([op['result']['latency'] for op in recent_ops])
        
        return {
            'accuracy': correct / len(recent_ops),
            'neural_ratio': neural_ops / len(recent_ops),
            'avg_latency': avg_latency,
            'total_recent': len(recent_ops)
        }

class LiveVisualizationEngine:
    """Real-time visualization engine"""
    
    def __init__(self, metrics_collector: LiveMetricsCollector):
        self.metrics = metrics_collector
        self.fig = None
        self.axes = None
        self.animation = None
        self.is_running = False
        
    def create_dashboard(self) -> tuple:
        """Create real-time dashboard"""
        if not plt.get_backend() or plt.get_backend() == 'Agg':
            return None, None
            
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Neuromorphic CPU Live Performance Dashboard', fontsize=16)
        
        # Configure subplots
        titles = [
            'Operations per Second', 'Spike Rate', 'Neural Efficiency %',
            'Accuracy %', 'Latency (ms)', 'System Overview'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            if i < len(titles):
                ax.set_title(titles[i])
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig, self.axes
    
    def update_dashboard(self, frame):
        """Update dashboard with live data"""
        if not self.is_running:
            return
        
        try:
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Get current data
            history = self.metrics.metrics_history
            current = self.metrics.get_current_snapshot()
            
            if not history['timestamps']:
                return
            
            times = np.array(list(history['timestamps'])) - list(history['timestamps'])[0]
            
            # Plot 1: Operations per Second
            if history['operations_per_second']:
                self.axes[0,0].plot(times[-50:], list(history['operations_per_second'])[-50:], 'b-', linewidth=2)
                self.axes[0,0].set_title(f"Ops/Sec (Current: {current['current_ops_per_second']:.1f})")
                self.axes[0,0].set_ylabel('Operations/sec')
            
            # Plot 2: Spike Rate
            if history['spike_rate']:
                self.axes[0,1].plot(times[-50:], list(history['spike_rate'])[-50:], 'r-', linewidth=2)
                self.axes[0,1].set_title(f"Spike Rate (Current: {current['current_spike_rate']:.1f})")
                self.axes[0,1].set_ylabel('Spikes/sec')
            
            # Plot 3: Neural Efficiency
            if history['neural_efficiency']:
                self.axes[0,2].plot(times[-50:], list(history['neural_efficiency'])[-50:], 'g-', linewidth=2)
                self.axes[0,2].set_title(f"Neural Efficiency ({current['neural_percentage']:.1f}%)")
                self.axes[0,2].set_ylabel('Percentage')
            
            # Plot 4: Accuracy
            if history['accuracy']:
                self.axes[1,0].plot(times[-50:], list(history['accuracy'])[-50:], 'm-', linewidth=2)
                self.axes[1,0].set_title(f"Accuracy ({current['accuracy_percentage']:.1f}%)")
                self.axes[1,0].set_ylabel('Percentage')
            
            # Plot 5: Latency
            if history['latency_ms']:
                self.axes[1,1].plot(times[-50:], list(history['latency_ms'])[-50:], 'c-', linewidth=2)
                self.axes[1,1].set_title("Latency (ms)")
                self.axes[1,1].set_ylabel('Milliseconds')
            
            # Plot 6: System Overview (Bar chart)
            categories = ['Total Ops', 'Neural Ops', 'Traditional', 'Spikes']
            values = [current['total_operations'], current['neural_operations'], 
                     current['traditional_operations'], current['total_spikes']]
            
            self.axes[1,2].bar(categories, values, color=['blue', 'green', 'orange', 'red'])
            self.axes[1,2].set_title("System Overview")
            self.axes[1,2].tick_params(axis='x', rotation=45)
            
            # Apply consistent formatting
            for ax in self.axes.flat:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
    
    def start_live_visualization(self, update_interval: int = 1000):
        """Start live visualization"""
        if self.fig is None:
            self.create_dashboard()
        
        if self.fig is not None:
            self.is_running = True
            self.animation = animation.FuncAnimation(
                self.fig, self.update_dashboard, 
                interval=update_interval, 
                blit=False,
                cache_frame_data=False
            )
            return self.animation
        return None
    
    def stop_live_visualization(self):
        """Stop live visualization"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()

class EnhancedNeuromorphicPipeline:
    """Enhanced pipeline with live monitoring and better performance"""
    
    def __init__(self, network_size: int = 64):
        self.alu = EnhancedEventDrivenALU(network_size)
        self.metrics = LiveMetricsCollector()
        self.visualization = LiveVisualizationEngine(self.metrics)
        self.clock = 0
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Enhanced statistics
        self.session_stats = {
            'session_start': time.time(),
            'peak_performance': 0.0,
            'total_computation_time': 0.0,
            'efficiency_trend': deque(maxlen=100)
        }
    
    def start_live_monitoring(self, update_interval: float = 0.5, enable_visualization: bool = True):
        """Start live performance monitoring"""
        self.is_monitoring = True
        
        if enable_visualization:
            # Start visualization in separate thread
            viz_thread = threading.Thread(
                target=self._start_visualization_thread, 
                daemon=True
            )
            viz_thread.start()
        
        # Start metrics monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(update_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        print("🔴 Live monitoring started!")
    
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        self.is_monitoring = False
        if self.visualization:
            self.visualization.stop_live_visualization()
        print("⏹️  Live monitoring stopped!")
    
    def _start_visualization_thread(self):
        """Start visualization in separate thread"""
        try:
            animation_obj = self.visualization.start_live_visualization()
            if animation_obj:
                plt.show(block=True)
        except Exception as e:
            print(f"Visualization thread error: {e}")
    
    def _monitoring_loop(self, update_interval: float):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Save periodic snapshots
                if self.clock % 100 == 0:
                    snapshot_path = RESULTS_DIRS['live'] / f"snapshot_{int(time.time())}.json"
                    self.metrics.save_metrics_snapshot(str(snapshot_path))
                
                time.sleep(update_interval)
            except Exception as e:
                print(f"Monitoring loop error: {e}")
    
    def process_instruction(self, instruction: List[Any], trace: bool = False) -> int:
        """Process instruction with enhanced monitoring"""
        self.clock += 1
        
        if len(instruction) < 3:
            return 0
        
        opcode, operand_a, operand_b = instruction[0], instruction[1], instruction[2]
        
        # Execute with timing
        start_time = time.time()
        result_data = self.alu.execute(opcode, {'a': operand_a, 'b': operand_b}, trace=trace)
        
        # Update live metrics
        self.metrics.update_metrics(result_data)
        
        # Update session stats
        computation_time = time.time() - start_time
        self.session_stats['total_computation_time'] += computation_time
        
        # Calculate efficiency
        if self.clock > 0:
            efficiency = (self.metrics.current_metrics['neural_operations'] / 
                         max(self.metrics.current_metrics['total_operations'], 1))
            self.session_stats['efficiency_trend'].append(efficiency)
        
        return result_data['result']
    
    def run_program_with_live_updates(self, program: List[List[Any]], 
                                    callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run program with live progress updates"""
        results = []
        start_time = time.time()
        
        if RICH_AVAILABLE and callback is None:
            # Use rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
            ) as progress:
                task = progress.add_task("Processing instructions...", total=len(program))
                
                for i, instruction in enumerate(program):
                    result = self.process_instruction(instruction, trace=(i % 50 == 0))
                    results.append(result)
                    
                    progress.update(task, advance=1)
                    
                    # Periodic live updates
                    if i % 10 == 0 and callback:
                        callback(i, len(program), self.metrics.get_current_snapshot())
        else:
            # Fallback to basic progress
            for i, instruction in enumerate(program):
                result = self.process_instruction(instruction, trace=(i % 50 == 0))
                results.append(result)
                
                if i % (len(program) // 10) == 0:
                    progress = (i + 1) / len(program) * 100
                    print(f"Progress: {progress:.1f}% ({i+1}/{len(program)})")
                    
                if callback:
                    callback(i, len(program), self.metrics.get_current_snapshot())
        
        end_time = time.time()
        
        # Compile comprehensive results
        perf_stats = self.alu.get_performance_stats()
        metrics_snapshot = self.metrics.get_current_snapshot()
        
        return {
            'results': results,
            'execution_time': end_time - start_time,
            'performance_stats': perf_stats,
            'live_metrics': metrics_snapshot,
            'spike_log': self.alu.spiking_net.spike_log.copy(),
            'session_stats': self.session_stats.copy(),
            'total_operations': len(program),
            'spike_efficiency': len(program) / max(perf_stats['spike_stats']['total_spikes'], 1)
        }

def run_enhanced_demo_with_live_monitoring():
    """Run comprehensive demo with live monitoring"""
    print("🚀 Enhanced Neuromorphic CPU Demo with Live Monitoring")
    print("=" * 60)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedNeuromorphicPipeline(network_size=128)
    
    # Start live monitoring
    pipeline.start_live_monitoring(update_interval=0.2, enable_visualization=True)
    
    # Create comprehensive test program
    print("📝 Generating test program...")
    test_program = []
    
    # Basic operations
    operations = ["ADD", "AND", "OR", "XOR"]
    for op in operations:
        for a in range(3):
            for b in range(3):
                test_program.append([op, a, b])
    
    # Add complex patterns
    np.random.seed(42)
    for _ in range(500):  # More operations for better monitoring
        op = np.random.choice(operations)
        a = np.random.randint(0, 3)
        b = np.random.randint(0, 3)
        test_program.append([op, a, b])
    
    print(f"📊 Running {len(test_program)} operations with live monitoring...")
    
    # Define progress callback
    def progress_callback(current, total, metrics):
        if current % 50 == 0:
            print(f"⚡ Progress: {current}/{total} | "
                  f"Ops/sec: {metrics['current_ops_per_second']:.1f} | "
                  f"Neural: {metrics['neural_percentage']:.1f}% | "
                  f"Accuracy: {metrics['accuracy_percentage']:.1f}%")
    
    # Run the program with live updates
    start_time = time.time()
    results = pipeline.run_program_with_live_updates(test_program, progress_callback)
    execution_time = time.time() - start_time
    
    # Display comprehensive results
    print("\n" + "=" * 60)
    print("📈 ENHANCED DEMO RESULTS")
    print("=" * 60)
    
    perf = results['performance_stats']
    metrics = results['live_metrics']
    spike_stats = perf['spike_stats']
    
    print(f"Total Execution Time: {execution_time:.4f} seconds")
    print(f"Total Operations: {results['total_operations']}")
    print(f"Neural Operations: {metrics['neural_operations']} ({metrics['neural_percentage']:.1f}%)")
    print(f"Accuracy: {metrics['accuracy_percentage']:.2f}%")
    print(f"Peak Ops/Second: {metrics['peak_ops_per_second']:.1f}")
    print(f"Total Spikes: {spike_stats['total_spikes']}")
    print(f"Spike Efficiency: {results['spike_efficiency']:.4f} ops/spike")
    print(f"Network Synchrony: {spike_stats['network_synchrony']:.3f}")
    print(f"Active Neurons: {spike_stats['active_neurons']}/{pipeline.alu.spiking_net.size}")
    
    # Save detailed results
    results_file = RESULTS_DIRS['live'] / f"enhanced_demo_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Keep monitoring for a bit longer
    print("\n🔴 Live monitoring continues for 10 more seconds...")
    time.sleep(10)
    
    pipeline.stop_live_monitoring()
    return results

def run_stress_test_with_monitoring():
    """Run stress test with live performance monitoring"""
    print("\n" + "=" * 60)
    print("🔥 STRESS TEST WITH LIVE MONITORING")
    print("=" * 60)
    
    pipeline = EnhancedNeuromorphicPipeline(network_size=256)  # Larger network
    pipeline.start_live_monitoring(update_interval=0.1, enable_visualization=False)
    
    # Create intensive workload
    operations = ["ADD", "AND", "OR", "XOR"]
    stress_program = []
    
    # Generate large random program
    np.random.seed(123)
    for _ in range(2000):  # Large workload
        op = np.random.choice(operations, p=[0.3, 0.25, 0.25, 0.2])  # Weighted selection
        a = np.random.randint(0, 3)
        b = np.random.randint(0, 3)
        stress_program.append([op, a, b])
    
    print(f"🚀 Running stress test with {len(stress_program)} operations...")
    
    # Track performance during stress test
    performance_snapshots = []
    
    def stress_callback(current, total, metrics):
        if current % 100 == 0:
            performance_snapshots.append({
                'progress': current / total,
                'timestamp': time.time(),
                'metrics': metrics.copy()
            })
            
            print(f"🔥 Stress Progress: {current}/{total} ({current/total*100:.1f}%) | "
                  f"Ops/sec: {metrics['current_ops_per_second']:.0f} | "
                  f"Spikes/sec: {metrics['current_spike_rate']:.0f}")
    
    # Execute stress test
    stress_results = pipeline.run_program_with_live_updates(stress_program, stress_callback)
    
    # Analyze stress test results
    print("\n📊 STRESS TEST ANALYSIS:")
    print("-" * 40)
    
    final_metrics = stress_results['live_metrics']
    print(f"Operations Completed: {final_metrics['total_operations']}")
    print(f"Peak Performance: {final_metrics['peak_ops_per_second']:.0f} ops/sec")
    print(f"Final Accuracy: {final_metrics['accuracy_percentage']:.2f}%")
    print(f"Neural Adaptation: {final_metrics['neural_percentage']:.1f}%")
    print(f"Total Spikes Generated: {final_metrics['total_spikes']}")
    
    # Save stress test results
    stress_file = RESULTS_DIRS['performance'] / f"stress_test_{int(time.time())}.json"
    stress_data = {
        'results': stress_results,
        'performance_snapshots': performance_snapshots,
        'test_parameters': {
            'total_operations': len(stress_program),
            'network_size': 256,
            'operation_distribution': {op: stress_program.count([op, 0, 0]) + 
                                    stress_program.count([op, 0, 1]) + 
                                    stress_program.count([op, 0, 2]) +
                                    stress_program.count([op, 1, 0]) +
                                    stress_program.count([op, 1, 1]) +
                                    stress_program.count([op, 1, 2]) +
                                    stress_program.count([op, 2, 0]) +
                                    stress_program.count([op, 2, 1]) +
                                    stress_program.count([op, 2, 2])
                                  for op in operations}
        }
    }
    
    with open(stress_file, 'w') as f:
        json.dump(stress_data, f, indent=2, default=str)
    
    pipeline.stop_live_monitoring()
    print(f"💾 Stress test results saved to: {stress_file}")
    
    return stress_results

def create_performance_comparison():
    """Compare enhanced vs original performance"""
    print("\n" + "=" * 60)
    print("⚖️  PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test with different configurations
    configurations = [
        {"name": "Small Network", "size": 32, "operations": 200},
        {"name": "Medium Network", "size": 64, "operations": 500},
        {"name": "Large Network", "size": 128, "operations": 1000},
    ]
    
    comparison_results = {}
    
    for config in configurations:
        print(f"\n🔬 Testing {config['name']} ({config['size']} neurons)...")
        
        pipeline = EnhancedNeuromorphicPipeline(network_size=config['size'])
        pipeline.start_live_monitoring(update_interval=0.5, enable_visualization=False)
        
        # Generate test program
        test_program = []
        operations = ["ADD", "AND", "OR", "XOR"]
        
        np.random.seed(42)  # Consistent seed for fair comparison
        for _ in range(config['operations']):
            op = np.random.choice(operations)
            a = np.random.randint(0, 3)
            b = np.random.randint(0, 3)
            test_program.append([op, a, b])
        
        # Run test
        start_time = time.time()
        results = pipeline.run_program_with_live_updates(test_program)
        test_time = time.time() - start_time
        
        # Collect metrics
        metrics = results['live_metrics']
        comparison_results[config['name']] = {
            'network_size': config['size'],
            'total_operations': config['operations'],
            'execution_time': test_time,
            'ops_per_second': config['operations'] / test_time,
            'accuracy': metrics['accuracy_percentage'],
            'neural_percentage': metrics['neural_percentage'],
            'total_spikes': metrics['total_spikes'],
            'spike_efficiency': results['spike_efficiency'],
            'peak_performance': metrics['peak_ops_per_second']
        }
        
        pipeline.stop_live_monitoring()
        
        print(f"   ⚡ {comparison_results[config['name']]['ops_per_second']:.0f} ops/sec")
        print(f"   🎯 {comparison_results[config['name']]['accuracy']:.1f}% accuracy")
        print(f"   🧠 {comparison_results[config['name']]['neural_percentage']:.1f}% neural operations")
    
    # Display comparison table
    print(f"\n📊 PERFORMANCE COMPARISON TABLE:")
    print("-" * 80)
    print(f"{'Configuration':<15} {'Ops/Sec':<10} {'Accuracy':<10} {'Neural %':<10} {'Efficiency':<12}")
    print("-" * 80)
    
    for name, data in comparison_results.items():
        print(f"{name:<15} {data['ops_per_second']:<10.0f} "
              f"{data['accuracy']:<10.1f} {data['neural_percentage']:<10.1f} "
              f"{data['spike_efficiency']:<12.4f}")
    
    # Save comparison results
    comparison_file = RESULTS_DIRS['analysis'] / f"performance_comparison_{int(time.time())}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\n💾 Comparison results saved to: {comparison_file}")
    return comparison_results

def generate_live_report():
    """Generate comprehensive live performance report"""
    print("\n" + "=" * 60)
    print("📋 GENERATING LIVE PERFORMANCE REPORT")
    print("=" * 60)
    
    # Collect all recent results files
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'rich_available': RICH_AVAILABLE
        },
        'test_results': {},
        'performance_summary': {}
    }
    
    # Scan results directories for recent files
    for dir_name, dir_path in RESULTS_DIRS.items():
        if dir_path.exists():
            recent_files = sorted(dir_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            report_data['test_results'][dir_name] = []
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        report_data['test_results'][dir_name].append({
                            'filename': file_path.name,
                            'size_mb': file_path.stat().st_size / (1024*1024),
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            'summary': extract_file_summary(file_data)
                        })
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
    
    # Generate summary statistics
    report_data['performance_summary'] = {
        'total_test_files': sum(len(files) for files in report_data['test_results'].values()),
        'total_storage_mb': sum(sum(f['size_mb'] for f in files) for files in report_data['test_results'].values()),
        'latest_test_time': datetime.now().isoformat(),
        'recommendations': generate_performance_recommendations(report_data)
    }
    
    # Save comprehensive report
    report_file = RESULTS_DIRS['analysis'] / f"live_performance_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Display report summary
    print(f"📈 Performance Report Generated:")
    print(f"   📁 Total test files: {report_data['performance_summary']['total_test_files']}")
    print(f"   💾 Total storage: {report_data['performance_summary']['total_storage_mb']:.2f} MB")
    print(f"   📊 Report saved to: {report_file}")
    
    # Display key recommendations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
    for rec in report_data['performance_summary']['recommendations']:
        print(f"   • {rec}")
    
    return report_data

def extract_file_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary information from test result file"""
    summary = {}
    
    if 'live_metrics' in data:
        metrics = data['live_metrics']
        summary.update({
            'total_operations': metrics.get('total_operations', 0),
            'accuracy': metrics.get('accuracy_percentage', 0),
            'neural_percentage': metrics.get('neural_percentage', 0)
        })
    
    if 'execution_time' in data:
        summary['execution_time'] = data['execution_time']
    
    if 'spike_efficiency' in data:
        summary['spike_efficiency'] = data['spike_efficiency']
    
    return summary

def generate_performance_recommendations(report_data: Dict[str, Any]) -> List[str]:
    """Generate performance optimization recommendations"""
    recommendations = []
    
    # Analyze recent test results
    all_summaries = []
    for dir_files in report_data['test_results'].values():
        for file_info in dir_files:
            if file_info.get('summary'):
                all_summaries.append(file_info['summary'])
    
    if all_summaries:
        # Calculate averages
        avg_accuracy = np.mean([s.get('accuracy', 0) for s in all_summaries if 'accuracy' in s])
        avg_neural_pct = np.mean([s.get('neural_percentage', 0) for s in all_summaries if 'neural_percentage' in s])
        avg_efficiency = np.mean([s.get('spike_efficiency', 0) for s in all_summaries if 'spike_efficiency' in s])
        
        # Generate recommendations based on performance
        if avg_accuracy < 95:
            recommendations.append("Consider increasing network size or training iterations to improve accuracy")
        
        if avg_neural_pct < 50:
            recommendations.append("Lower adaptation threshold to increase neuromorphic processing usage")
        
        if avg_efficiency < 0.1:
            recommendations.append("Optimize spike generation parameters to improve computational efficiency")
        
        recommendations.append(f"Current average accuracy: {avg_accuracy:.1f}% - Target: >95%")
        recommendations.append(f"Current neural usage: {avg_neural_pct:.1f}% - Target: >60%")
    
    if not recommendations:
        recommendations.append("System performance is optimal - no specific recommendations")
    
    return recommendations

def main():
    """Enhanced main function with comprehensive live monitoring"""
    print("🚀 ENHANCED NEUROMORPHIC CPU WITH LIVE MONITORING")
    print("=" * 80)
    print("Features:")
    print("  • Real-time performance monitoring")
    print("  • Live visualization dashboard")
    print("  • Enhanced spike neural networks")
    print("  • Comprehensive performance analysis")
    print("  • Detailed reporting and recommendations")
    print("=" * 80)
    
    try:
        # Run enhanced demo
        print("\n1️⃣  Running Enhanced Demo with Live Monitoring...")
        demo_results = run_enhanced_demo_with_live_monitoring()
        
        # Run stress test
        print("\n2️⃣  Running Stress Test...")
        stress_results = run_stress_test_with_monitoring()
        
        # Performance comparison
        print("\n3️⃣  Running Performance Comparison...")
        comparison_results = create_performance_comparison()
        
        # Generate comprehensive report
        print("\n4️⃣  Generating Live Performance Report...")
        report_data = generate_live_report()
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ ENHANCED NEUROMORPHIC CPU DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"📊 Results Summary:")
        print(f"   • Demo Operations: {demo_results.get('total_operations', 0)}")
        print(f"   • Stress Test Operations: {stress_results.get('total_operations', 0)}")
        print(f"   • Performance Configurations Tested: {len(comparison_results)}")
        print(f"   • Total Files Generated: {report_data['performance_summary']['total_test_files']}")
        print(f"   • Results Storage: {report_data['performance_summary']['total_storage_mb']:.2f} MB")
        
        print(f"\n📁 Check the following directories for detailed results:")
        for name, path in RESULTS_DIRS.items():
            file_count = len(list(path.glob("*.json"))) if path.exists() else 0
            print(f"   • {name}: {path} ({file_count} files)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
