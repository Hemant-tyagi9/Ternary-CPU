TNEG, TZERO, TPOS = -1, 0, 1
MODEL_DIR = "models"

class TernarySignal:
    def __init__(self, value):
        # Convert unbalanced to balanced automatically
        if value in [0, 1, 2]:
            value = value - 1  # Maps 0→-1, 1→0, 2→+1
        if value not in [-1, 0, 1]:
            raise ValueError(f"Invalid ternary signal: {value}")
        self.value = value

    def __str__(self):
        return str(self.value)

def int_to_ternary(n, digits=5):
    """Convert integer to ternary representation"""
    if n < 0:
        raise ValueError("Only non-negative integers supported")

    base3 = []
    for _ in range(digits):
        n, r = divmod(n, 3)
        base3.append(r - 1)  # Convert to -1, 0, 1 representation
    return base3[::-1]

def ternary_to_int(t):
    """Convert ternary to integer"""
    return sum((v + 1) * (3 ** i) for i, v in enumerate(reversed(t)))

def display_ternary(t):
    """Display ternary in readable format"""
    symbols = {-1: '-', 0: '0', 1: '+'}
    return ''.join(symbols[x] for x in t)

def one_hot_encode(y, num_classes=3):
    encoded = np.zeros((len(y), num_classes))
    for i, val in enumerate(y):
        encoded[i, val] = 1
    return encoded

# Gate functions
def ternary_and(a, b): return min(a, b)
def ternary_or(a, b): return max(a, b)
def ternary_xor(a, b): return (a - b) % 3
def ternary_nand(a, b): return 2 - min(a, b)
def ternary_nor(a, b): return 2 - max(a, b)

