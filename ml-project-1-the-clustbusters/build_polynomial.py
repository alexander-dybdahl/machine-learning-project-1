import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """

    N = x.shape[0]

    phi = np.zeros((N, degree+1))

    for n in range(N):
        phi[n] = [x[n]**j for j in range(degree+1)]
    
    return phi


def build_poly_multivariate(tx, degree):
    """
    Generate polynomial basis function for multivariate input data tx, for j=0 up to j=degree.
    
    Args:
        tx: numpy array of shape (N, D), N is the number of samples, D is the number of features.
        degree: integer, the degree of the polynomial.
        
    Returns:
        poly: numpy array of shape (N, D*(degree+1) - degree)
              For each feature, it contains the powers of the feature from 0 up to degree.
    """
    
    N, D = tx.shape
    if degree == 1:
        return tx
    
    # Identify the continuous columns (assuming you've standardized, one-hot columns will be around -1 and 1)
    continuous_cols = [i for i in range(D) if np.abs(tx[:, i].min()) > 1e-3 and np.abs(tx[:, i].max()) > 1e-3]
    
    poly_list = [tx]  # Start with the original features
    
    for d in range(2, degree + 1):  # Start from 2 since the original data acts as the first degree
        for feature in continuous_cols:
            poly_list.append((tx[:, feature] ** d).reshape(-1, 1))
            
    poly = np.hstack(poly_list)
    
    return poly