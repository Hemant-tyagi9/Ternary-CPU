import numpy as np
from typing import List, Dict, Tuple

class SpikingNeuralNetwork:
    """Core spiking neural network implementation"""
    
    def __init__(self, num_neurons: int = 64):
        self.num_neurons = num_neurons
        self.neurons = np.zeros(num_neurons)
        self.thresholds = np.random.uniform(0.5, 1.5, num_neurons)
        self.weights = np.random.normal(0, 0.1, (num_neurons, num_neurons))
        self.ternary_map = {-1: 0, 0: 1, 1: 2}  # Maps balanced to unbalanced
        
    def reset(self):
        """Reset network state"""
        self.neurons = np.zeros(self.num_neurons)
        
    def spike(self, inputs: List[int]) -> Dict[int, int]:
        """
        Process inputs through spiking network
        Returns dict of {neuron_index: ternary_output}
        """
        # Convert ternary inputs to spikes
        spike_input = np.array([self._to_spike(x) for x in inputs])
        
        # Leaky integrate-and-fire dynamics
        self.neurons = 0.9 * self.neurons + np.dot(spike_input, self.weights)
        
        # Generate spikes
        spikes = (self.neurons > self.thresholds).astype(int)
        self.neurons[spikes == 1] = 0  # Reset spiked neurons
        
        # Convert spikes to ternary outputs
        return {
            i: self._to_ternary(np.random.random())
            for i, s in enumerate(spikes) if s == 1
        }
    
    def _to_spike(self, ternary_val: int) -> float:
        """Convert ternary value to spike input"""
        return {-1: -0.8, 0: 0.1, 1: 0.8}.get(ternary_val, 0)
    
    def _to_ternary(self, val: float) -> int:
        """Convert random value to ternary output"""
        if val < 0.3: return -1
        if val < 0.7: return 0
        return 1
