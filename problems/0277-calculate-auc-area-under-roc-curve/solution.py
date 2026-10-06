import numpy as np

def calculate_auc(y_true, y_scores) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) using the trapezoidal rule.
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # 1. Reuse the ROC curve point generation logic
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)
    
    if num_positives == 0 or num_negatives == 0:
        raise ValueError("y_true must contain both 0 and 1 classes.")
        
    desc_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_indices]
    y_true_sorted = y_true[desc_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores_sorted) != 0)[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    
    tps = np.cumsum(y_true_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_true_sorted == 0)[threshold_indices]
    
    # Generate FPR (x-coordinates) and TPR (y-coordinates)
    fpr = np.r_[0.0, fps / num_negatives]
    tpr = np.r_[0.0, tps / num_positives]
    
    # 2. Integrate using the Trapezoidal Rule
    # The width of each trapezoid along the X-axis (FPR)
    # np.diff(fpr) gives x_i - x_{i-1}
    dt = np.diff(fpr)
    
    # The average height of each trapezoid along the Y-axis (TPR)
    # (y_i + y_{i-1}) / 2
    avg_height = (tpr[1:] + tpr[:-1]) / 2.0
    
    # Total area is the sum of (width * average height)
    auc = np.sum(dt * avg_height)
    
    return float(auc)
