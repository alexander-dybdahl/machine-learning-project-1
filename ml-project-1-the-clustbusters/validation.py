import numpy as np


def compute_accuracy(y, y_pred):
    TP = np.sum((y == 1) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    return (TP+TN)/y.shape[0]


def compute_f1_score(y_true, y_pred):
    """
    Calculate the F1 score.
    
    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    
    Returns:
    - float: F1 score.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def compute_type_2_error(y_true, y_pred):
    """
    Calculate the Type II error rate (false negative rate).

    Parameters:
    - y_true: The true labels
    - y_pred: The predicted labels

    Returns:
    - type_2_error_rate: The Type II error rate
    """
    # Count the number of false negatives
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Count the number of true positives
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # Calculate the Type II error rate
    type_2_error_rate = FN / (FN + TP)
    
    return type_2_error_rate


def confusion_matrix(y_true, y_pred):

    """Compute confusion matrix for binary classification."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TP, FP], [FN, TN]])