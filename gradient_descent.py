import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss using MSE."""

    N = y.shape[0]

    return 1/(2*N) * np.linalg.norm(y - tx @ w)**2


def compute_gradient(y, tx, w):
    """Computes the gradient at w."""

    N = y.shape[0]

    e = y - tx @ w
    grad = - 1/N * tx.T @ e
    
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    """The Gradient Descent (GD) algorithm."""
    
    w = initial_w
    
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)

        w = w - gamma * grad

        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=compute_mse(y, tx, w)))
            # converge criterion

        if np.linalg.norm(grad) < tol:
            print("Gradient close to zero")
            break

        if n_iter == max_iters-1:
            print("Reached max iterations")

    mse = compute_mse(y, tx, w)

    return w, mse
