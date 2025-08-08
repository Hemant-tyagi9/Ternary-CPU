import numpy as np
import time
from typing import List, Dict, Callable, Optional
from collections import deque
import threading

class TernarySpike:
    """Represents a ternary spike event"""
    def __init__(self, value: int, timestamp: float, source_id: int):
        self.value = value % 3  # Ensure ternary
        self.timestamp = timestamp
        self.source_id = source_id
    
    def __repr__(self):
        return f"Spike(val={self.value}, t={self.timestamp:.4f}, src={self.source_id})"

class TernarySpikingNeuron:
    """Brain-inspired spiking neuron with ternary values"""
    
    def __init__(self, neuron_id: int, threshold: float = 1.0, 
                 decay_rate: float = 0.95, refractory_period: float = 0.001):
        self.neuron_id = neuron_id
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.refractory_period = refractory_period
        
        # Neuromorphic state variables
        self.membrane_potential = 0.0
        self.last_spike_time = 0.0
        self.is_refractory = False
        
        # Synaptic connections
        self.synapses: Dict[int, float] = {}  # source_id -> weight
        self.spike_buffer = deque(maxlen=1000)
        
        # Event-driven processing
        self.output_connections: List[Callable] = []
        
    def add_synapse(self, source_id: int, weight: float):
        """Add synaptic connection from another neuron"""
        self.synapses[source_id] = weight
    
    def connect_output(self, callback: Callable):
        """Connect this neuron's output to another component"""
        self.output_connections.append(callback)
    
    def receive_spike(self, spike: TernarySpike):
        """Process incoming spike (event-driven)"""
        current_time = time.time()
        
        # Check if in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return
        
        # Apply membrane potential decay
        time_since_last = current_time - self.last_spike_time
        self.membrane_potential *= (self.decay_rate ** time_since_last)
        
        # Add synaptic input if connection exists
        if spike.source_id in self.synapses:
            weight = self.synapses[spike.source_id]
            # Ternary-weighted input
            input_current = (spike.value * weight) / 2.0  # Normalize to [0,1]
            self.membrane_potential += input_current
            
            # Check for spike generation
            if self.membrane_potential >= self.threshold:
                self.generate_spike(current_time)
    
    def generate_spike(self, timestamp: float):
        """Generate output spike when threshold is reached"""
        # Determine ternary output value based on membrane potential
        if self.membrane_potential >= 2 * self.threshold:
            spike_value = 2
        elif self.membrane_potential >= 1.5 * self.threshold:
            spike_value = 1
        else:
            spike_value = 0
        
        # Create and propagate spike
        output_spike = TernarySpike(spike_value, timestamp, self.neuron_id)
        
        # Reset neuron state
        self.membrane_potential = 0.0
        self.last_spike_time = timestamp
        
        # Send to all connected components
        for callback in self.output_connections:
            callback(output_spike)
        
        # Store in buffer for analysis
        self.spike_buffer.append(output_spike)

class EventDrivenTernaryALU:
    """Event-driven ALU using spiking ternary neurons"""
    
    def __init__(self):
        self.neurons: Dict[str, List[TernarySpikingNeuron]] = {}
        self.operation_results = deque(maxlen=100)
        self.setup_operations()
    
    def setup_operations(self):
        """Setup neural circuits for each operation"""
        operations = ["ADD", "SUB", "AND", "OR", "XOR"]
        
        for op in operations:
            # Create a small network for each operation
            self.neurons[op] = [
                TernarySpikingNeuron(i, threshold=0.8) 
                for i in range(4)  # 2 input + 1 hidden + 1 output
            ]
            
            # Wire the network
            input1, input2, hidden, output = self.neurons[op]
            
            # Connect inputs to hidden layer
            hidden.add_synapse(input1.neuron_id, 0.5)
            hidden.add_synapse(input2.neuron_id, 0.5)
            
            # Connect hidden to output
            output.add_synapse(hidden.neuron_id, 1.0)
            
            # Setup output collection
            output.connect_output(
                lambda spike, op=op: self.collect_result(op, spike)
            )
    
    def collect_result(self, operation: str, spike: TernarySpike):
        """Collect operation results"""
        self.operation_results.append({
            'operation': operation,
            'result': spike.value,
            'timestamp': spike.timestamp
        })
    
    def execute_async(self, operation: str, a: int, b: int):
        """Execute operation asynchronously using spiking neurons"""
        if operation not in self.neurons:
            raise ValueError(f"Operation {operation} not supported")
        
        current_time = time.time()
        
        # Create input spikes
        spike_a = TernarySpike(a, current_time, 0)
        spike_b = TernarySpike(b, current_time + 0.0001, 1)
        
        # Send spikes to appropriate neurons
        input1, input2, hidden, output = self.neurons[operation]
        
        # Simulate spike propagation
        threading.Thread(target=lambda: [
            input1.receive_spike(spike_a),
            time.sleep(0.001),  # Synaptic delay
            input2.receive_spike(spike_b)
        ]).start()
    
    def get_latest_result(self, timeout: float = 0.01) -> Optional[Dict]:
        """Get the latest computation result"""
        start_time = time.time()
        initial_count = len(self.operation_results)
        
        while time.time() - start_time < timeout:
            if len(self.operation_results) > initial_count:
                return self.operation_results[-1]
            time.sleep(0.0001)
        
        return None

class NeuromorphicTernaryCPU:
    """True neuromorphic CPU with event-driven processing"""
    
    def __init__(self):
        self.memory = {}  # Event-driven memory
        self.registers = {}
        self.spiking_alu = EventDrivenTernaryALU()
        self.event_queue = deque()
        self.running = False
        
        # Neuromorphic characteristics
        self.energy_consumption = 0.0
        self.spike_count = 0
        
    def start_processing(self):
        """Start asynchronous event processing"""
        self.running = True
        processing_thread = threading.Thread(target=self._process_events)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_events(self):
        """Continuously process events (brain-like)"""
        while self.running:
            if self.event_queue:
                event = self.event_queue.popleft()
                self._handle_event(event)
                self.energy_consumption += 0.001  # Minimal energy per event
            time.sleep(0.0001)  # Minimal processing delay
    
    def _handle_event(self, event):
        """Handle individual events"""
        if event['type'] == 'ALU_OPERATION':
            self.spiking_alu.execute_async(
                event['operation'], 
                event['operand1'], 
                event['operand2']
            )
            self.spike_count += 2  # Input spikes
    
    def execute_instruction_async(self, operation: str, a: int, b: int):
        """Execute instruction asynchronously (neuromorphic style)"""
        event = {
            'type': 'ALU_OPERATION',
            'operation': operation,
            'operand1': a,
            'operand2': b,
            'timestamp': time.time()
        }
        self.event_queue.append(event)
    
    def get_metrics(self):
        """Get neuromorphic performance metrics"""
        return {
            'energy_consumption': self.energy_consumption,
            'spike_count': self.spike_count,
            'events_processed': len(self.event_queue),
            'efficiency': self.spike_count / max(1, self.energy_consumption)
        }
    
    def stop_processing(self):
        """Stop event processing"""
        self.running = False

# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Neuromorphic Ternary CPU")
    
    # Create and start neuromorphic CPU
    cpu = NeuromorphicTernaryCPU()
    cpu.start_processing()
    
    # Execute some operations asynchronously
    operations = [
        ("ADD", 1, 2),
        ("AND", 2, 1),
        ("OR", 0, 2),
        ("XOR", 1, 1)
    ]
    
    print("Executing operations asynchronously...")
    for op, a, b in operations:
        cpu.execute_instruction_async(op, a, b)
        print(f"Queued: {op}({a}, {b})")
    
    # Wait for results
    time.sleep(0.1)
    
    # Check results
    for _ in range(10):  # Try to get results
        result = cpu.spiking_alu.get_latest_result()
        if result:
            print(f"Result: {result['operation']} = {result['result']} "
                  f"(t={result['timestamp']:.6f})")
    
    # Show neuromorphic metrics
    metrics = cpu.get_metrics()
    print(f"\n📊 Neuromorphic Metrics:")
    print(f"Energy consumption: {metrics['energy_consumption']:.6f}")
    print(f"Spike count: {metrics['spike_count']}")
    print(f"Efficiency: {metrics['efficiency']:.2f}")
    
    cpu.stop_processing()
    print("✅ Neuromorphic simulation complete!")
