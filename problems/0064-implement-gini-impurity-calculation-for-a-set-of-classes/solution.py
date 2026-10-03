import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list or array of class labels.

    :param y: List or array of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    if len(y) == 0:
        return 0.0

    # Count occurrences of each unique class
    _, counts = np.unique(y, return_counts=True)

    # Compute probabilities p_i = count / total_samples
    probabilities = counts / len(y)

    # Gini Impurity = 1 - sum(p_i^2)
    gini = 1.0 - np.sum(probabilities**2)

    return round(float(gini), 3)