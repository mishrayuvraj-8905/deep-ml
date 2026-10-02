import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Build the Vandermonde design matrix with increasing powers (x^0, x^1, ..., x^d)
    X = np.vander(x, degree + 1, increasing=True)
    
    # Solve the least-squares problem: X * c = y
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    return coeffs.tolist()