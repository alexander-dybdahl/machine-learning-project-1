import numpy as np
from helpers import batch_iter


def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    N = y.shape[0]

    return 1/(2*N) * np.linalg.norm(y - tx @ w)**2


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B = y.shape[0]

    e = y - tx @ w
    grad = - 1/B * tx.T @ e
    
    return grad


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1, tol=1e-8):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w
    
    N = y.shape[0]
    B = N // batch_size

    for n_iter in range(max_iters):
        
        grad_sum = np.zeros(tx.shape[1])

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, B):
            grad_sum += compute_stoch_gradient(minibatch_y, minibatch_tx, w)

        g = 1/B * grad_sum

        w = w - gamma * g
        
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=compute_mse(y, tx, w)))
            # converge criterion

        if np.linalg.norm(g) < tol:
            print("Gradient close to zero")
            break

        if n_iter == max_iters-1:
            print("Reached max iterations")
        
    mse = compute_mse(y, tx, w)

    return w, mse