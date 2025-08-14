class TernaryMemory:
    def __init__(self, size=27):
        self.memory = [None] * size  # Store raw values
        
    def load(self, address):
        if 0 <= address < len(self.memory):
            return self.memory[address]
        return None
        
    def store(self, address, value):
        if 0 <= address < len(self.memory):
            # Store the value directly without modification
            self.memory[address] = value
