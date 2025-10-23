import numpy as np


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
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = -np.mean(y * safe_log(pred) + (1 - y) * safe_log(1 - pred))
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
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
