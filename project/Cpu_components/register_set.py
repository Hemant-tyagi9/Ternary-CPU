class RegisterSet:
    def __init__(self, num_registers=9):
        self.registers = [0] * num_registers

    def read(self, index):
        return self.registers[index % len(self.registers)]

    def write(self, index, value):
        self.registers[index % len(self.registers)] = value % 3
