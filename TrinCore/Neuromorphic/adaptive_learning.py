import numpy as np
from typing import Dict
from .spiking_neurons import SpikingNeuralNetwork

class AdaptiveLearner:
    """Implements various neuromorphic learning rules"""
    
    def __init__(self, network: SpikingNeuralNetwork):
        self.network = network
        self.trace = np.zeros_like(network.weights)
        
    def stdp_update(self, pre_spikes: Dict[int, int], post_spikes: Dict[int, int]):
        """
        Spike-timing dependent plasticity
        pre_spikes: {neuron_idx: spike_time}
        post_spikes: {neuron_idx: spike_time}
        """
        for pre_idx, pre_time in pre_spikes.items():
            for post_idx, post_time in post_spikes.items():
                delta_t = post_time - pre_time
                if delta_t > 0:  # Causal (pre before post)
                    self.network.weights[post_idx, pre_idx] += 0.1 * np.exp(-delta_t/10)
                else:  # Anti-causal
                    self.network.weights[post_idx, pre_idx] -= 0.1 * np.exp(delta_t/10)
        
        # Normalize weights
        self.network.weights = np.clip(self.network.weights, -1, 1)
        
    def reward_modulated_stdp(self, reward: float):
        """Reward-modulated learning rule"""
        self.network.weights += 0.01 * reward * np.outer(
            self.network.neurons, 
            self.network.neurons
        )
        self.network.weights = np.clip(self.network.weights, -1, 1)
