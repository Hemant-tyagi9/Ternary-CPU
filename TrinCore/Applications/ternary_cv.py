from typing import Dict, List
import numpy as np
from NN.nn_integration import NeuralIntegration

class TernaryImageProcessor:
    """Ternary computer vision operations"""
    
    def __init__(self, use_neural=True):
        self.neural = use_neural
        if use_neural:
            self.alu = NeuralIntegration()
            self.alu.train_models(["AND", "OR", "XOR"])
    
    def ternary_threshold(self, image):
        """Convert image to ternary values"""
        thresholds = [85, 170]
        return np.digitize(image, thresholds)
    
    def apply_operation(self, image1, image2, operation):
        """Apply ternary operation to two images"""
        if self.neural:
            result = np.zeros_like(image1)
            for i in range(image1.shape[0]):
                for j in range(image1.shape[1]):
                    result[i,j] = self.alu.execute_operation(
                        operation, 
                        image1[i,j], 
                        image2[i,j]
                    )
            return result
        else:
            op_fn = {
                "AND": ternary_and,
                "OR": ternary_or,
                "XOR": ternary_xor
            }.get(operation, lambda a,b: 0)
            return np.vectorize(op_fn)(image1, image2)
