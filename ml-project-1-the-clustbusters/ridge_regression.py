import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    N = y.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    mse = 1/(2*N) * np.linalg.norm(y - tx @ w)**2

    return w, mse