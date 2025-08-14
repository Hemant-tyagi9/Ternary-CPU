class ProgramCounter:
    """Program Counter for instruction sequencing"""
    def __init__(self):
        self.value = 0
        self.max_value = 242  # 3^5 - 1 for 5-trit addressing

    def increment(self):
        self.value = (self.value + 1) % (self.max_value + 1)

    def jump(self, address):
        if 0 <= address <= self.max_value:
            self.value = address
        else:
            raise ValueError(f"Address {address} out of range [0, {self.max_value}]")

    def get(self):
        return self.value

    def reset(self):
        self.value = 0
