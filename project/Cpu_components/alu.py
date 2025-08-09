class TernaryALU:
    def __init__(self):
        self.flags = {"zero": False, "carry": False}

    def execute(self, opcode, a, b):
        a %= 3
        b %= 3
        if opcode == "ADD":
            result = (a + b) % 3
        elif opcode == "SUB":
            result = (a - b) % 3
        elif opcode == "AND":
            result = min(a, b)
        elif opcode == "OR":
            result = max(a, b)
        elif opcode == "XOR":
            result = (a - b) % 3 if a != b else 0
        else:
            result = 0

        self.flags["zero"] = result == 0
        self.flags["carry"] = (a + b) >= 3 if opcode == "ADD" else (a - b) < 0
        return result


