import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    N = y.shape[0]
    mse = 1/(2*N) * np.linalg.norm(y - tx @ w)**2

    return w, mse
