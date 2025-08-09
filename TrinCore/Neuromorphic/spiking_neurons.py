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

    def spike(self, spike_input):
        spike_input = np.array(spike_input, dtype=float)
        
        # If input size doesn't match weight matrix, pad or truncate
        input_size = self.weights.shape[0]
        if spike_input.shape[0] != input_size:
            padded = np.zeros(input_size)
            padded[:min(spike_input.shape[0], input_size)] = spike_input[:min(spike_input.shape[0], input_size)]
            spike_input = padded
        
        self.neurons = 0.9 * self.neurons + np.dot(spike_input, self.weights)
        
        # Use self.thresholds if it's an array, or fall back to a default
        if hasattr(self, "thresholds"):
            threshold = self.thresholds
        elif hasattr(self, "threshold"):
            threshold = self.threshold
        else:
            threshold = 1.0  # default value
        
        spikes = self.neurons > threshold
        self.neurons[spikes] = 0
        return spikes

    
    def _to_spike(self, ternary_val: int) -> float:
        """Convert ternary value to spike input"""
        return {-1: -0.8, 0: 0.1, 1: 0.8}.get(ternary_val, 0)
    
    def _to_ternary(self, val: float) -> int:
        """Convert random value to ternary output"""
        if val < 0.3: return -1
        if val < 0.7: return 0
        return 1
