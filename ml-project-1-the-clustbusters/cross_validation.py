import numpy as np
from ridge_regression import *
from gradient_descent import *
from plots import *
from logistic_regression import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_loss(y, tx, method, k_indices, k, lambda_, loss_type, args=()):
    """return the loss of method for a fold corresponding to k_indices"""
    
    test_indices = k_indices[k]
    train_indices = np.ravel(np.delete(k_indices, k, axis=0))

    tx_te, y_te = tx[test_indices], y[test_indices]
    tx_tr, y_tr = tx[train_indices], y[train_indices]

    w, mse = method(y_tr, tx_tr, lambda_, *args)

    if loss_type.lower()=="reg":
        loss_tr = np.sqrt(2 * mse)
        loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    elif loss_type.lower()=="class":
        loss_tr = np.sqrt(2 * calculate_loss(y_tr, tx_tr, w))
        loss_te = np.sqrt(2 * calculate_loss(y_te, tx_te, w))
    elif loss_type.lower()=="weighted":
        w_pos = len(y) / (2 * len(y[y == 1]))
        w_neg = len(y) / (2 * len(y[y == 0]))
        loss_tr = np.sqrt(2 * weighted_log_loss(y_tr, tx_tr, w, w_pos, w_neg))
        loss_te = np.sqrt(2 * weighted_log_loss(y_te, tx_te, w, w_pos, w_neg))

    return loss_tr, loss_te


def cross_validation_lambda(y, tx, method, k_fold, lambdas, seed=1, loss_type="reg", args=()):
    """cross validation over regularisation parameter lambda."""
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        print("lambda", lambda_)
        loss_tr, loss_te = zip(*(cross_validation_loss(y, tx, method, k_indices, k, lambda_, loss_type, args=args) for k in range(k_fold)))
        rmse_tr.append(np.mean(loss_tr))
        rmse_te.append(np.mean(loss_te))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

    best_loss = min(rmse_te)
    best_lambda = lambdas[np.argmin(rmse_te)]

    return best_lambda, best_loss


# def cross_validation_f1(y, tx, train_func, predict_func, args=None, k_fold=5, seed=1, cutoff=0.5):
#     """ Perform cross-validation for classification focusing on F1 score. """
    
#     # Generate k-fold indices
#     k_indices = build_k_indices(y, k_fold, seed)
    
#     # Lists to store F1 scores for each fold
#     f1_list = []
    
#     for k in range(k_fold):
#         # Split data
#         validation_idx = k_indices[k]
#         train_idx = [idx for sublist in np.delete(k_indices, k, axis=0) for idx in sublist]
        
#         tx_train = tx[train_idx]
#         y_train = y[train_idx]
#         tx_validation = tx[validation_idx]
#         y_validation = y[validation_idx]
        
#         # Train the model
#         w, _ = train_func(y_train, tx_train, *args)  # Assuming train_func returns weights
        
#         # Validate the model
#         y_pred = predict_func(tx_validation, w, cutoff)
        
#         # Calculate and store F1 score for this fold
#         f1_list.append(compute_f1_score(y_validation, y_pred))
    
#     # Calculate average F1 score across folds
#     avg_f1 = np.mean(f1_list)
    
#     return avg_f1


# def cross_validation_cutoffs(y, tx, train_func, predict_func, cutoffs, args=None, k_fold=5, seed=1):
#     """Perform cross-validation for different cutoff values focusing on F1 score."""
    
#     best_cutoff = None
#     best_f1 = float("-inf")
    
#     # Lists to store average F1 scores for plotting
#     f1_scores = []
    
#     for cutoff in cutoffs:
#         avg_f1 = cross_validation_f1(y, tx, train_func, predict_func, args=args, k_fold=k_fold, seed=seed, cutoff=cutoff)
        
#         f1_scores.append(avg_f1)
        
#         if avg_f1 > best_f1:
#             best_f1 = avg_f1
#             best_cutoff = cutoff
    
#     # Plot F1 scores against cutoff values
#     plot_f1_scores(cutoffs, f1_scores)
    
#     return best_cutoff


def validate_f1(y, tx, w, predict_func, cutoff=0.5):
    """ Validate a model given weights for classification focusing on F1 score. """
    
    y_pred = predict_func(tx, w, cutoff)
    return compute_f1_score(y, y_pred)


def cross_validation_f1(y, tx, w, predict_func, k_fold=5, seed=1, cutoff=0.5):
    """ Perform cross-validation for classification focusing on F1 score using given weights. """
    
    # Generate k-fold indices
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Lists to store F1 scores for each fold
    f1_list = []
    
    for k in range(k_fold):
        # Split data for validation
        validation_idx = k_indices[k]
        tx_validation = tx[validation_idx]
        y_validation = y[validation_idx]
        
        # Validate the model using given weights
        f1_list.append(validate_f1(y_validation, tx_validation, w, predict_func, cutoff))
    
    # Calculate average F1 score across folds
    avg_f1 = np.mean(f1_list)
    
    return avg_f1


def cross_validation_cutoffs(y, tx, w, predict_func, cutoffs, k_fold=5, seed=1, plot=True):
    """Perform cross-validation for different cutoff values focusing on F1 score using given weights."""
    
    best_cutoff = None
    best_f1 = float("-inf")
    
    # Lists to store average F1 scores for plotting
    f1_scores = []
    
    for cutoff in cutoffs:
        avg_f1 = cross_validation_f1(y, tx, w, predict_func, k_fold=k_fold, seed=seed, cutoff=cutoff)
        
        f1_scores.append(avg_f1)
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_cutoff = cutoff
    
    # Plot F1 scores against cutoff values
    if plot:
        plot_f1_scores(cutoffs, f1_scores)
    
    return best_cutoff, f1_scores


def plot_f1_scores(cutoffs, f1_scores):
    """Plot F1 scores against cutoff values."""
    plt.figure(figsize=(10, 6))
    plt.plot(cutoffs, f1_scores, label='F1 Score', color='blue', marker='o')
    plt.xlabel('Cutoff Value')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Cutoff Values')
    plt.legend()
    plt.grid(True)
    plt.savefig("cross_validation_f1")