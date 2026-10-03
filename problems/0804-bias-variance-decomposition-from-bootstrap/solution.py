import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    preds = np.asarray(predictions)
    y = np.asarray(y_true)

    # Expected prediction for each test point across the B models
    mean_preds = np.mean(preds, axis=0)

    # 1. Bias squared: mean squared difference between the average prediction and true value
    bias_squared = np.mean((mean_preds - y) ** 2)

    # 2. Variance: mean population variance of predictions across models for each test point (ddof=0)
    variance = np.mean(np.var(preds, axis=0, ddof=0))

    # 3. Mean Squared Error: mean squared difference between all predictions and true values
    mse = np.mean((preds - y[np.newaxis, :]) ** 2)

    return {
        'bias_squared': float(bias_squared),
        'variance': float(variance),
        'mse': float(mse)
    }