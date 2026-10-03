import numpy as np


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[np.ndarray, list[float]]:
    """Gradient-descent training algorithm for Softmax regression.

    Optimizing parameters with Cross Entropy loss.

    Parameters:
    - X: Feature matrix of shape (m, n)
    - y: Target class labels of shape (m,)
    - learning_rate: Learning rate for gradient descent
    - iterations: Number of iterations to train

    Returns:
    - coeffs: Learned weight matrix of shape (k, n + 1) where column 0
              is the bias term followed by feature weights.
    - losses: List of unnormalized cross-entropy loss per iteration.
    """
    m, n = X.shape
    k = len(np.unique(y))

    # Prepend bias column of ones as the FIRST column
    X_b = np.hstack([np.ones((m, 1)), X])

    # One-hot encode labels to shape (m, k)
    Y_onehot = np.zeros((m, k))
    Y_onehot[np.arange(m), y] = 1.0

    # Initialize coefficients of shape (k, n + 1)
    coeffs = np.zeros((k, n + 1))
    losses = []

    for _ in range(iterations):
        # Linear logits: (m, n + 1) @ (n + 1, k) -> (m, k)
        Z = np.dot(X_b, coeffs.T)

        # Numerically stable Softmax
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # Cross-entropy loss (unnormalized sum over samples)
        loss = -np.sum(Y_onehot * np.log(P + 1e-15))
        losses.append(loss)

        # Gradient without 1/m normalization: dL/d(coeffs) of shape (k, n + 1)
        grad = np.dot((P - Y_onehot).T, X_b)

        # Gradient descent parameter update
        coeffs -= learning_rate * grad

    return coeffs, losses