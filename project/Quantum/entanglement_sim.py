import numpy as np

class TernaryQubit:
    """Simulation of a ternary quantum bit (qutrit)"""
    
    def __init__(self):
        self.state = np.array([1, 0, 0], dtype=complex)  # |0> state
    
    def x_gate(self):
        """Ternary X gate (cyclic shift)"""
        self.state = np.roll(self.state, 1)
    
    def z_gate(self):
        """Ternary Z gate (phase shift)"""
        phases = [1, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)]
        self.state *= phases
    
    def hadamard(self):
        """Ternary Hadamard-like gate"""
        w = np.exp(2j*np.pi/3)
        h_matrix = np.array([
            [1, 1, 1],
            [1, w, w**2],
            [1, w**2, w]
        ]) / np.sqrt(3)
        self.state = h_matrix @ self.state
    
    def measure(self):
        """Measure the qutrit"""
        probabilities = np.abs(self.state)**2
        return np.random.choice(3, p=probabilities)
