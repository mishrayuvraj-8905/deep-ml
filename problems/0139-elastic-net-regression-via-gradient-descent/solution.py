import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        error = X @ weights + bias - y

        # MSE gradient + L1 subgradient + L2 gradient (bias is not regularized)
        grad_w = (X.T @ error) / n_samples + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        grad_b = np.sum(error) / n_samples

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias