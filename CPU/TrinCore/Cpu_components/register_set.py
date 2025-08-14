class Register:
    def __init__(self, value=0):
        self.value = value % 3  # Ensure ternary value

class RegisterSet:
    def __init__(self, num_registers=9):
        self.registers = [0] * num_registers  # Using simple list for performance
    
    def read(self, index):
        return self.registers[index % len(self.registers)]
    
    def write(self, index, value):
        self.registers[index % len(self.registers)] = value % 3
