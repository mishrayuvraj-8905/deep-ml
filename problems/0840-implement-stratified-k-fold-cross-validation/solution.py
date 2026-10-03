import numpy as np


def stratified_kfold_indices(y, n_splits):
    """Generate train/test indices for stratified K-fold cross-validation.

    Args:
        y: 1D array-like of integer class labels
        n_splits: number of folds

    Returns:
        A list of [train_indices, test_indices] pairs, one per fold.
    """
    y = np.asarray(y)
    n_samples = len(y)

    # Preserve class order of appearance in y
    _, unique_idx = np.unique(y, return_index=True)
    classes = y[np.sort(unique_idx)]

    test_folds = [[] for _ in range(n_splits)]

    # 1. Split each class's indices into contiguous chunks across folds
    for c in classes:
        cls_indices = np.where(y == c)[0]
        n_cls = len(cls_indices)

        base_size = n_cls // n_splits
        remainder = n_cls % n_splits

        start = 0
        for fold in range(n_splits):
            fold_size = base_size + (1 if fold < remainder else 0)
            test_folds[fold].extend(cls_indices[start : start + fold_size])
            start += fold_size

    # 2. Build train/test index lists as standard Python lists
    splits = []
    all_indices = set(range(n_samples))

    for fold in range(n_splits):
        test_idx = test_folds[fold]
        # Train indices are sorted complement of test indices
        train_idx = sorted(all_indices - set(test_idx))

        splits.append([train_idx, test_idx])

    return splits
	