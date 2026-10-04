import numpy as np

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
    """
    Compute the polynomial kernel between two vectors.
    """
    return float((gamma * np.dot(x, y) + coef0) ** degree)