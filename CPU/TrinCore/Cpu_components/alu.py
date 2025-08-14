class TernaryALU:
    def __init__(self):
        self.flags = {
            "zero": False,
            "carry": False,
            "negative": False,
            "overflow": False
        }
        self.operation_count = 0
        
    def _update_flags(self, result, a=None, b=None):
        result = result % 3
        self.flags["zero"] = (result == 0)
        self.flags["negative"] = (result == 2)
        return result
        
    def execute(self, op, a, b=None):
        a = a % 3
        if b is not None:
            b = b % 3
            
        op = op.upper()
        self.operation_count += 1
        
        # Optimized operation implementations
        if op == "ADD":
            result = (a + b) % 3
        elif op == "SUB":
            result = (a - b) % 3
        elif op == "AND":
            result = min(a, b)
        elif op == "OR":
            result = max(a, b)
        elif op == "XOR":
            result = (a - b) % 3 if a != b else 0
        elif op == "NAND":
            result = 2 - min(a, b)
        elif op == "NOR":
            result = 2 - max(a, b)
        elif op == "NOT":
            result = 2 - a
        else:
            result = 0
            
        return self._update_flags(result, a, b)
