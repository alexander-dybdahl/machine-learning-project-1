import numpy as np
from cross_validation import *

def svm_gradient_descent(y, tx, initial_w, C=1.0, learning_rate=0.001, epochs=1000, threshold=0, tol=1e-8):
    # Initialize weights and bias
    w = initial_w
    b = 0
    
    for epoch in range(epochs):
        for i in range(tx.shape[0]):
            if y[i] * (np.dot(tx[i], w) + b) < 1:
                w -= learning_rate * (-C * y[i] * tx[i] + w)
            else:
                w -= learning_rate * w
                
    predictions = predict_svm(tx, w, b, threshold)
    loss = np.mean((y - predictions)**2)

    return w, b, loss


def predict_svm(x, w, b, threshold=0):
    return np.where(x @ w + b > threshold, 1, 0)


def cross_validation_svm(y, tx, k_fold, seed, C_values, learning_rates, thresholds):
    # Building k indices
    k_indices = build_k_indices(y, k_fold, seed)
    max_f1 = -float('inf')  # set initial max F1 score to negative infinity
    best_C = None
    best_lr = None
    best_threshold = None

    # Loop through all hyperparameters
    for C in C_values:
        for lr in learning_rates:
            for threshold in thresholds:
                f1s = []  # average F1 score for this hyperparameter combination
                print(f"Testing C: {C}, lr: {lr}, threshold: {threshold}")
                for k in range(k_fold):
                    # Splitting data into training and validation sets
                    test_indices = k_indices[k]
                    train_indices = k_indices[~k].ravel()
                    y_train, y_test = y[train_indices], y[test_indices]
                    tx_train, tx_test = tx[train_indices], tx[test_indices]
                    
                    # Training SVM model
                    initial_w = np.ones(tx.shape[1])
                    w, b, _ = svm_gradient_descent(y_train, tx_train, initial_w, C, lr, epochs=2)
                    
                    # Predicting using the trained SVM model
                    pred = predict_svm(tx_test, w, b, threshold)
                    
                    # Calculating F1 score
                    f1 = compute_f1_score(y_test, pred)
                    if f1 == 0. or f1 == 1.:
                        print("Extreme F1 value", f1)
                    else:
                        f1s.append(f1)
                
                # Calculating average F1 score
                f1_avg = np.mean(f1s)
                
                print(f"Result : {f1_avg}")
                
                # Updating best hyperparameters if current combination has higher F1 score
                if f1_avg > max_f1:
                    max_f1 = f1_avg
                    best_C = C
                    best_lr = lr
                    best_threshold = threshold

    return best_C, best_lr, best_threshold, max_f1