import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features) or list of lists
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Ensure X is a NumPy float array
    X = np.asarray(X, dtype=float)
    n_samples, _ = X.shape
    
    # 1. Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Sample covariance matrix
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # 3. Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    # 4. Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Clip tiny negative eigenvalues caused by numerical precision issues
    sorted_eigenvalues = np.maximum(sorted_eigenvalues, 0.0)
    
    # 5. Explained variance ratio
    total_var = np.sum(sorted_eigenvalues)
    if total_var == 0:
        return [0.0] * len(sorted_eigenvalues)
        
    variance_ratio = sorted_eigenvalues / total_var
    
    return variance_ratio.tolist()
                