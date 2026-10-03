import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    # Ensure float numpy array to handle list inputs safely
    X = np.asarray(X, dtype=float)
    
    # 1. Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    # eigh is optimized and stable for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Select top n_components eigenvectors (largest eigenvalues)
    top_indices = np.argsort(eigenvalues)[::-1][:n_components]
    V_k = eigenvectors[:, top_indices]  # Shape: (n_features, n_components)
    
    # 5. Project centered data into low-dimensional space: Z = X_centered @ V_k
    Z = np.dot(X_centered, V_k)
    
    # 6. Reconstruct back into original space: X_hat = Z @ V_k.T + mean
    X_reconstructed = np.dot(Z, V_k.T) + mean
    
    # 7. Compute Mean Squared Error (MSE)
    mse = np.mean((X - X_reconstructed) ** 2)
    
    return float(mse)