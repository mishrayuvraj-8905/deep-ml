import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    """
    # 1. Standardize features (zero mean, unit variance)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero if a feature has zero variance
    std[std == 0] = 1.0
    data_standardized = (data - mean) / std
    
    # 2. Compute covariance matrix of standardized data
    cov_matrix = np.cov(data_standardized, rowvar=False)
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    top_components = eigenvectors[:, top_k_indices]
    
    # 5. Enforce deterministic sign (first non-zero element positive)
    for col in range(k):
        vec = top_components[:, col]
        non_zero_indices = np.nonzero(vec)[0]
        if len(non_zero_indices) > 0:
            first_val = vec[non_zero_indices[0]]
            if first_val < 0:
                top_components[:, col] *= -1
                
    # 6. Round to 4 decimal places
    return np.round(top_components, 4)
    