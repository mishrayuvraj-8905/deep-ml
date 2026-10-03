from itertools import product
import numpy as np


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict,
    model_fn: callable,
    scoring_fn: callable,
) -> tuple:
  """Perform grid search to find optimal hyperparameters.

  Args:
      X_train: Training features
      y_train: Training labels
      X_val: Validation features
      y_val: Validation labels
      param_grid: Dict mapping parameter names to lists of values to try
      model_fn: Function(X_train, y_train, X_val, **params) -> predictions
      scoring_fn: Function(y_true, y_pred) -> score (higher is better)

  Returns:
      Tuple of (best_params dict, best_score rounded to 4 decimals)
  """
  keys = list(param_grid.keys())
  value_lists = list(param_grid.values())

  best_params = None
  best_score = -float("inf")

  # Generate Cartesian product of all hyperparameter candidate lists
  for combination in product(*value_lists):
    params = dict(zip(keys, combination))

    # Train model and generate predictions on validation data
    y_pred = model_fn(X_train, y_train, X_val, **params)

    # Evaluate validation score
    score = scoring_fn(y_val, y_pred)

    if score > best_score:
      best_score = score
      best_params = params

  return best_params, round(float(best_score), 4)