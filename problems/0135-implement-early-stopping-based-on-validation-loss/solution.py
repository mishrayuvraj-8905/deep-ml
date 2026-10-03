from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine the stop epoch and best epoch based on validation losses, patience, and min_delta.
    
    Args:
        val_losses: List of validation losses per epoch.
        patience: Number of epochs to wait for an improvement before stopping.
        min_delta: Minimum change in loss to qualify as an improvement.
        
    Returns:
        Tuple of (stop_epoch, best_epoch)
    """
    if not val_losses:
        return -1, -1
        
    best_loss = val_losses[0]
    best_epoch = 0
    patience_counter = 0
    stop_epoch = len(val_losses) - 1
    
    for epoch in range(1, len(val_losses)):
        loss = val_losses[epoch]
        
        # Check if loss improved by more than min_delta
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_epoch = epoch
                break
                
    return stop_epoch, best_epoch