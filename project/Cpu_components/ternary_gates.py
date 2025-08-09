def ternary_and(a, b):
    return min(a, b)

def ternary_or(a, b):
    return max(a, b)

def ternary_xor(a, b):
    return (a - b) % 3 if a != b else 0

def ternary_nand(a, b):
    return 2 - min(a, b)

def ternary_nor(a, b):
    return 2 - max(a, b)

# Ternary NOT gate (unary)
def ternary_not(a):
    return 2 - a

# Ternary IMPLY gate (a -> b)
def ternary_imply(a, b):
    if a <= b:
        return 2
    else:
        return b

# Ternary EQUIVALENCE gate (a <-> b)
def ternary_eq(a, b):
    if a == b:
        return 2
    else:
        return 0

# Ternary Majority gate (for three inputs)
def ternary_majority(a, b, c):
    if (a + b + c) >= 2:
        return 2
    else:
        return 0

# Ternary MUX (Multiplexer)
def ternary_mux(select, a, b):
    if select == 0:
        return a
    elif select == 1:
        return b
    else:
        # A common interpretation for the third state
        return min(a, b)
