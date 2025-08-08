class TernaryMemory:
    def __init__(self, size=27):
        self.memory = [0] * size

    def load(self, address):
        return self.memory[address % len(self.memory)]

    def store(self, address, value):
        self.memory[address % len(self.memory)] = value % 3
