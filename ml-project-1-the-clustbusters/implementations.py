import numpy as np
from helpers import *




# Least Squares

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


# Gradient Decent

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



# Stochastic Gradient Decent

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


# Ridge Regression

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    N = y.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
    mse = 1/(2*N) * np.linalg.norm(y - tx @ w)**2

    return w, mse


# Logistic Regression

def sigmoid(t):
    """apply sigmoid function on t."""
    res = np.where(t < -20, 0, t)
    res = np.where(t > 20, 1, res)
    return 1 / (1 + np.exp(-res))


def safe_log(x, minval=1e-15):
    return np.log(np.clip(x, minval, 1.0 - minval))


def weighted_log_loss(y, tx, w, w_pos, w_neg):
    """Calculate the weighted log loss."""
    y_pred = sigmoid(tx.dot(w))
    loss = -np.mean(w_pos * y * np.log(y_pred) + w_neg * (1 - y) * np.log(1 - y_pred))
    return loss


def calculate_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = -np.mean(y * safe_log(pred) + (1 - y) * safe_log(1 - pred))
    return loss


def calculate_gradient(y, tx, w):
    """Compute the gradient of loss."""
    grad = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]
    return grad


def logistic_regression(y, tx, initial_w=None, max_iters=1000, gamma=0.1, threshold=1e-4, seed=1):
    if initial_w is None:
        np.random.seed(seed)
        initial_w = np.random.random(tx.shape[1])
    w = initial_w

    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=calculate_loss(y, tx, w)))
        
        if np.linalg.norm(grad) < threshold:
            print("Gradient close to zero")
            break
            
        if iter == max_iters-1:
            print("Reached max iterations")

    mse = np.mean((y - sigmoid(tx @ w))**2) / 2
    return w, mse


# Regularized Logistic Regression

def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters=1000, gamma=0.1, threshold=1e-4, seed=1):
    if initial_w is None:
        np.random.seed(seed)
        initial_w = np.random.random(tx.shape[1])
    w = initial_w

    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=calculate_loss(y, tx, w)))
        
        if np.linalg.norm(grad) < threshold:
            print("Gradient close to zero")
            break
            
        if iter == max_iters-1:
            print("Reached max iterations")

    mse = np.mean((y - sigmoid(tx @ w))**2) / 2
    return w, mse


# Predicting

def predict_reg(x, w, cutoff=0.5):
    return np.where(x @ w > cutoff, 1, 0)

def predict_class(x, w, cutoff=0.5):
    return np.where(sigmoid(x @ w) > cutoff, 1, 0)