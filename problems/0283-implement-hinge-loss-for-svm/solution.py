import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average hinge loss for SVM classification.
    """
    loss = np.maximum(0, 1 - y_true * y_pred)
    return round(np.mean(loss), 4)