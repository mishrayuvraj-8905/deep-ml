import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    # Ensure inputs are numpy arrays and flatten 1D feature arrays properly
    X_train = np.asarray(X_train).ravel()
    y_train = np.asarray(y_train).ravel()
    X_val = np.asarray(X_val).ravel()
    y_val = np.asarray(y_val).ravel()
    
    def build_design_matrix(x_data, d):
        return np.column_stack([x_data**i for i in range(d + 1)])
    
    train_errors = []
    val_errors = []
    
    Phi_val = build_design_matrix(X_val, degree)
    
    for n in train_sizes:
        X_n = X_train[:n]
        y_n = y_train[:n]
        
        Phi_train = build_design_matrix(X_n, degree)
        
        # Compute least-squares weights using the Moore-Penrose pseudoinverse
        w = np.linalg.pinv(Phi_train) @ y_n
        
        # Training MSE (cast to standard Python float)
        y_pred_train = Phi_train @ w
        train_mse = float(np.mean((y_pred_train - y_n) ** 2))
        train_errors.append(train_mse)
        
        # Validation MSE (cast to standard Python float)
        y_pred_val = Phi_val @ w
        val_mse = float(np.mean((y_pred_val - y_val) ** 2))
        val_errors.append(val_mse)
        
    # Diagnosis at the largest training size (n*)
    e_train_star = train_errors[-1]
    e_val_star = val_errors[-1]
    
    if e_train_star > bias_threshold:
        diagnosis = "high_bias"
    elif (e_train_star <= bias_threshold) and (e_val_star - e_train_star > variance_threshold):
        diagnosis = "high_variance"
    else:
        diagnosis = "good_fit"
        
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnosis
    }