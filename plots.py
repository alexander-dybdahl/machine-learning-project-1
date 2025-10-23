import matplotlib.pyplot as plt
import numpy as np
from preproccessing import *
from validation import *

def print_mse(names, mses):
    
    # Header
    header = f"{'Method':<41} {'MSE Value':<15}"
    print(header)
    print("-" * len(header))

    # Content
    for name, mse in zip(names, mses):
        print(f"{name:<40} {mse:.10f}")


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    # plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_lambda")


def plot_pca(tx):
    """
    Apply PCA on tx and plot the cumulative variance explained by the principal components.

    Parameters:
    - tx: original data
    """
    tx = np.copy(tx)
    tx = median_imputation(tx)
    tx, similar_kept_columns = remove_similar_data(tx, threshold=0.01)
    tx, correlation_kept_columns = remove_correlation_data(tx, threshold=0.99)
    
    # Step 1: Standardize the dataset
    mean = np.mean(tx, axis=0)
    std = np.std(tx, axis=0)
    tx_standardized = (tx - mean) / std
    
    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(tx_standardized, rowvar=False)
    
    # Step 3: Obtain the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort eigenvalues in decreasing order
    sorted_eigenvalues = eigenvalues[::-1]
    
    # Step 5: Compute the explained variance and cumulative explained variance
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Step 6: Plot the cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Principal Components')
    plt.show()

    
def plot_results(y, predictions, names):
    header = f"{'Method':<58} {'Accuracy':<10} {'F1 Score':<10} {'Type II Error':<15}"
    print(header)
    print("-" * len(header))

    # Content
    for name, pred in zip(names, predictions):
        accuracy = compute_accuracy(y, pred)
        f1 = compute_f1_score(y, pred)
        type_2_error = compute_type_2_error(y, pred)
        print(f"{name:<60} {accuracy:.3f}      {f1:.3f}      {type_2_error:.3f}")
