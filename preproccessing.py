import numpy as np


def mean_imputation(tx, tx_train):
    for col in range(tx.shape[1]):
        column_mean = np.nanmean(tx[:, col])
        tx[np.isnan(tx[:, col]), col] = column_mean
        tx_train[np.isnan(tx_train[:, col]), col] = column_mean
    return tx, tx_train


def median_imputation(tx):
    for col in range(tx.shape[1]):
        column_median = np.nanmedian(tx[:, col])
        tx[np.isnan(tx[:, col]), col] = column_median
    return tx


def train_test_split(tx, y, test_size=0.2, seed=None):
    """
    Splits the data into training and test sets.

    Parameters:
    - tx: Feature data.
    - y: Labels.
    - test_size: Proportion of the dataset to include in the test split. Default is 0.2.
    - random_seed: Seed for the random number generator.

    Returns:
    - tx_train, tx_test, y_train, y_test: Split datasets.
    """
    tx = np.copy(tx)
    y = np.copy(y)

    # Ensure tx and y have matching number of samples
    assert tx.shape[0] == y.shape[0], "Mismatched number of samples between tx and y."

    # Setting the random seed for reproducibility
    np.random.seed(seed)

    # Generate random indices for splitting
    shuffled_indices = np.random.permutation(len(y))
    test_set_size = int(len(y) * test_size)
    
    # Splitting indices for training and test sets
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Create training and test datasets
    tx_train = tx[train_indices]
    y_train = y[train_indices]
    tx_test = tx[test_indices]
    y_test = y[test_indices]
    
    return tx_train, tx_test, y_train, y_test


def select_random_data(tx, y, fraction=0.1, seed=1):

    num_samples = tx.shape[0]
    np.random.seed(seed)
    random_indices = np.random.choice(num_samples, size=int(num_samples * fraction), replace=False)
    return tx[random_indices, :], y[random_indices]


def apply_pca(tx, tx_test, n_components):
    """
    Apply PCA on tx and keep the top n_components.
    
    Parameters:
    - tx: original data
    - n_components: number of top principal components to keep
    
    Returns:
    - tx_pca: data projected onto the top n_components principal components
    """
    
    # Step 1: Standardize the dataset
    mean = np.mean(tx, axis=0)
    std = np.std(tx, axis=0)
    tx_standardized = (tx - mean) / std
    
    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(tx_standardized, rowvar=False)
    
    # Step 3: Obtain the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Step 4: Sort eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    
    # Step 5: Project the train data
    tx_pca = tx @ top_eigenvectors

    # Step 6: Project the test data
    tx_test_standarized = (tx_test - mean) / std
    tx_test_pca = tx_test_standarized @ top_eigenvectors

    return tx_pca, tx_test_pca


def one_hot_encode(tx, min_unique, tx_test=None):
    encoded_data = []
    encoded_data_test = []
    categories_list = []  # List to keep track of unique values per column
    continuous_data = []
    continuous_data_test = []
    
    for col in range(tx.shape[1]):
        unique_vals = np.unique(tx[:, col])
        
        if len(unique_vals) <= min_unique:
            categories_list.append(unique_vals)
            
            for val in unique_vals:
                encoded_col = (tx[:, col] == val).astype(int)
                encoded_data.append(encoded_col)
                
                if tx_test is not None:
                    encoded_col_test = (tx_test[:, col] == val).astype(int)
                    encoded_data_test.append(encoded_col_test)
        else:
            continuous_data.append(tx[:, col])
            
            if tx_test is not None:
                continuous_data_test.append(tx_test[:, col])
    
    encoded_data = np.column_stack(encoded_data + continuous_data)
    
    if tx_test is not None:
        encoded_data_test = np.column_stack(encoded_data_test + continuous_data_test)
        return encoded_data, encoded_data_test
    
    return encoded_data,


def remove_similar_data(tx, threshold=0.01):
    low_variance_cols = np.where(np.var(tx, axis=0) < threshold)[0]
    kept_columns = np.delete(np.arange(tx.shape[1]), list(low_variance_cols))
    return tx[:, kept_columns], kept_columns


def remove_correlation_data(tx, threshold=0.90):
    correlation_matrix = np.corrcoef(tx, rowvar=False)
    columns_to_remove = set()
    
    # Only consider upper triangle of the correlation matrix to avoid duplicates
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            if np.abs(correlation_matrix[i, j]) > threshold:
                columns_to_remove.add(j)
    
    kept_columns = np.delete(np.arange(tx.shape[1]), list(columns_to_remove))
    return tx[:, kept_columns], kept_columns


def  remove_missing_data_columns(tx, threshold=0.7):

    N = tx.shape[0]
    missing_counts = np.sum(np.isnan(tx), axis=0)  # Count missing values for each column
    kept_columns = np.where(missing_counts / N < threshold)[0]  # Find columns below the threshold

    return tx[:, kept_columns], kept_columns


def standardize_data(tx):
    means = np.nanmean(tx, axis=0)
    stds = np.nanstd(tx, axis=0)
    
    # Avoid division by zero
    stds[stds == 0] = 1.0
    
    tx_standardized = (tx - means) / stds
    return tx_standardized


def undersample_negatives(y, X, ratio=0.5):
    """
    Undersample the negative class to achieve the desired ratio of positives to negatives.
    
    Args:
    - y (np.array): A 1D numpy array of labels.
    - X (np.array): A 2D numpy array of features.
    - ratio (float): Desired ratio of positive (class 1) samples in the resulting data. Default is 0.5.
    
    Returns:
    - y_undersampled (np.array): A 1D numpy array of labels after undersampling.
    - X_undersampled (np.array): A 2D numpy array of features after undersampling.
    """
    
    # Indices for each class
    idx_positives = np.where(y == 1)[0]
    idx_negatives = np.where(y == 0)[0]
    
    # Calculate the number of negative samples to retain
    num_required_negatives = int(len(idx_positives) / ratio - len(idx_positives))
    
    # Randomly sample negative indices to match the desired count
    idx_negatives_downsampled = np.random.choice(idx_negatives, size=num_required_negatives, replace=False)
    
    # Combine and shuffle indices
    idx_undersampled = np.concatenate([idx_positives, idx_negatives_downsampled])
    np.random.shuffle(idx_undersampled)
    
    # Extract balanced dataset
    y_undersampled = y[idx_undersampled]
    X_undersampled = X[idx_undersampled]
    
    return y_undersampled, X_undersampled


def random_oversample(y, X):
    """Randomly oversample minority class to balance the dataset."""
    
    # Find indices of positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Calculate how many samples to add
    num_to_add = len(neg_indices) - len(pos_indices)
    
    # Randomly sample instances to duplicate
    duplicate_indices = np.random.choice(pos_indices, size=num_to_add, replace=True)
    
    # Duplicate these instances
    X_new_samples = X[duplicate_indices]
    y_new_samples = y[duplicate_indices]
    
    # Append these to the original data
    X_oversampled = np.vstack((X, X_new_samples))
    y_oversampled = np.hstack((y, y_new_samples))
    
    # Shuffle the dataset to mix the original and oversampled instances
    shuffle_indices = np.arange(len(y_oversampled))
    np.random.shuffle(shuffle_indices)
    
    X_oversampled = X_oversampled[shuffle_indices]
    y_oversampled = y_oversampled[shuffle_indices]
    
    return y_oversampled, X_oversampled


def preprocess_data(data_train, data_test, y_train, min_unique, var_tol, corr_tol, empty_tol, num_components, ratio, validation_size, seed, pca=False, hon=False):
    
    y_train[y_train == -1] = 0
    
    if hon:
        tx_train_encoded, tx_test_encoded = one_hot_encode(data_train, min_unique, data_test)
        print("encoding", tx_train_encoded.shape, tx_test_encoded.shape)

    # Impute missing data
    tx_train = median_imputation(tx_train_encoded)
    tx_test = median_imputation(tx_test_encoded)
    print("imputing", tx_train.shape, tx_test.shape)
    
    # Remove similar data based on variance
    tx_train, similar_kept_columns = remove_similar_data(tx_train, threshold=var_tol)
    tx_test = tx_test[:, similar_kept_columns]
    print("removing low var", tx_train.shape, tx_test.shape)
    
    # Remove correlated data
    tx_train, correlation_kept_columns = remove_correlation_data(tx_train, threshold=corr_tol)
    tx_test = tx_test[:, correlation_kept_columns]
    print("removing high corr", tx_train.shape, tx_test.shape)
    
    if pca:
        tx_train, tx_test = apply_pca(tx_train, tx_test, num_components)
        print("pca", tx_train.shape, tx_test.shape)

    # Standardize
    tx_test = standardize_data(tx_test)
    tx_train = standardize_data(tx_train)
    print("standardize", tx_train.shape, tx_test.shape)

    # Splitting the training data into training and validation
    tx_train, tx_val, y_train, y_val = train_test_split(tx_train, y_train, test_size=validation_size, seed=seed)
    print("splitting", tx_train.shape, tx_test.shape, tx_val.shape)

    # # Undersampling and Oversampling
    y_train, tx_train = undersample_negatives(y_train, tx_train, ratio)
    y_train, tx_train = random_oversample(y_train, tx_train)
    print("balancing", tx_train.shape, tx_test.shape, tx_val.shape)
    

    return tx_train, tx_val, tx_test, y_train, y_val
