import numpy as np
from typing import Union
from typing import Dict, List, Any

def ternary_to_binary(value: int, bits: int = 2) -> str:
    """Convert ternary value to binary representation"""
    if value == 0:
        return '0' * bits
    elif value == 1:
        return '0' + '1'*(bits-1)
    elif value == 2:
        return '1' * bits
    raise ValueError("Ternary value must be 0, 1, or 2")

def binary_to_ternary(binary: str) -> int:
    """Convert binary string to ternary value"""
    if binary == '00':
        return 0
    elif binary == '01':
        return 1
    elif binary == '11':
        return 2
    raise ValueError("Invalid binary ternary representation")

def pack_ternary(values: List[int], width: int = 9) -> int:
    """Pack multiple ternary values into a single integer"""
    packed = 0
    for i, val in enumerate(values[:width]):
        packed += val * (3 ** i)
    return packed

def unpack_ternary(packed: int, width: int = 9) -> List[int]:
    """Unpack a packed integer into ternary values"""
    values = []
    for _ in range(width):
        values.append(packed % 3)
        packed = packed // 3
    return values

def ternary_checksum(data: Union[List[int], np.ndarray]) -> int:
    """Calculate ternary checksum"""
    return sum(x % 3 for x in data) % 3

def normalize_ternary(values: np.ndarray) -> np.ndarray:
    """Normalize values to proper ternary (0,1,2)"""
    return np.clip(np.round(values), 0, 2).astype(int)

def ternary_hamming_distance(a: int, b: int) -> int:
    """Calculate Hamming distance between ternary values"""
    a = normalize_ternary(np.array([a]))[0]
    b = normalize_ternary(np.array([b]))[0]
    return abs(a - b)

def generate_ternary_matrix(rows: int, cols: int) -> np.ndarray:
    """Generate random ternary matrix"""
    return np.random.randint(0, 3, size=(rows, cols))def ternary_to_decimal(trits):
    return sum([val * (3 ** i) for i, val in enumerate(reversed(trits))])

def decimal_to_ternary(n, length=3):
    trits = []
    for _ in range(length):
        trits.insert(0, n % 3)
        n //= 3
    return trits
