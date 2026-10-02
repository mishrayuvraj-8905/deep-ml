import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    for _ in range(max_iter):
        # 1. Predict
        y_pred = X @ weights + bias
        error = y_pred - y

        # 2. Gradient step on MSE (smooth part only)
        grad_w = (X.T @ error) / n_samples
        grad_b = np.sum(error) / n_samples
        w_temp = weights - learning_rate * grad_w

        # 3. Proximal step: soft-threshold with lr * alpha
        w_new = soft_threshold(w_temp, learning_rate * alpha)

        # 4. Bias update (no regularization)
        bias = bias - learning_rate * grad_b

        # 5. Convergence check on weight change
        converged = np.max(np.abs(w_new - weights)) < tol
        weights = w_new
        if converged:
            break

    return weights, bias


    
    # Your code here
    
