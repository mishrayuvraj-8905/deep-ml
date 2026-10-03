import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    if k <= 1:
        raise ValueError("k must be greater than 1.")
    if k > n_samples:
        raise ValueError("k cannot be greater than the number of samples.")

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    # Calculate fold sizes, distributing remainder samples evenly across early folds
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    splits: List[Tuple[List[int], List[int]]] = []
    current = 0

    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        splits.append((train_idx.tolist(), test_idx.tolist()))
        current = end

    return splits