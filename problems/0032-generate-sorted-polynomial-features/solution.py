import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    """
    Generates all polynomial feature combinations up to the given degree inclusive,
    and sorts the resulting features for each sample from lowest to highest value.

    Args:
        X: 2-D NumPy array of shape (n_samples, n_features)
        degree: Non-negative integer representing the maximum polynomial degree

    Returns:
        A new 2-D NumPy array with ascending-sorted polynomial features per sample.
    """
    n_samples, n_features = X.shape
    
    # 1. Generate all column index combinations for degrees from 0 to degree
    combos = []
    for d in range(degree + 1):
        combos.extend(combinations_with_replacement(range(n_features), d))
        
    output = []
    
    # 2. Compute feature values for each sample row
    for row in X:
        features = []
        for combo in combos:
            # Degree 0 represents the bias/constant term (value of 1)
            # Higher degrees multiply the selected feature columns together
            val = np.prod([row[i] for i in combo]) if combo else 1.0
            features.append(val)
            
        # 3. Sort features for the current sample from lowest to highest value
        features.sort()
        output.append(features)
        
    return np.array(output)
    