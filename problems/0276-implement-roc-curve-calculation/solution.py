import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Convert inputs to numpy arrays for efficient vector operations
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Total number of actual positives and actual negatives
    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)
    
    # Handle edge case where one class is completely missing
    if num_positives == 0 or num_negatives == 0:
        raise ValueError("y_true must contain both 0 and 1 classes.")
        
    # Sort scores and their corresponding true labels in descending order
    desc_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_indices]
    y_true_sorted = y_true[desc_indices]
    
    # Find unique thresholds to avoid redundant calculations at identical scores
    # We look for where the score changes
    distinct_value_indices = np.where(np.diff(y_scores_sorted) != 0)[0]
    
    # Include the last element index to capture the final threshold
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    # Calculate cumulative True Positives (TP) and False Positives (FP)
    # as the threshold moves from highest score to lowest score
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    
    # Filter TP and FP values only at distinct threshold changes
    tps = tps[threshold_indices]
    fps = fps[threshold_indices]
    
    # Calculate rates
    tpr = tps / num_positives
    fpr = fps / num_negatives
    
    # ROC curves traditionally start at (FPR=0, TPR=0) 
    # This corresponds to a threshold higher than the maximum score
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    
    return list(fpr), list(tpr)
