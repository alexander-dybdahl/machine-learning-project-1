# Machine Learning Mathematical Implementations

This repository contains from-scratch implementations of fundamental machine learning algorithms, focusing on the mathematical foundations and numerical optimization techniques.

## Core Mathematical Algorithms

### 1. Least Squares Regression
**Mathematical Foundation**: Solves the normal equation $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

```python
w = np.linalg.solve(tx.T @ tx, tx.T @ y)
```

- **Closed-form solution**: Direct matrix inversion approach
- **Loss function**: $L(\mathbf{w}) = \frac{1}{2N}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$
- **Optimal**: Minimizes squared error analytically

### 2. Gradient Descent Optimization
**Mathematical Foundation**: Iterative optimization using $\mathbf{w}_{t+1} = \mathbf{w}_t - \gamma \nabla L(\mathbf{w}_t)$

```python
grad = -1/N * tx.T @ (y - tx @ w)  # ∇L(w) = -1/N * X^T(y - Xw)
w = w - gamma * grad
```

- **Gradient computation**: $\nabla L(\mathbf{w}) = -\frac{1}{N}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w})$
- **Learning rate**: Controls step size $\gamma$
- **Convergence**: Stops when $\|\nabla L(\mathbf{w})\| < \text{tolerance}$

### 3. Stochastic Gradient Descent (SGD)
**Mathematical Foundation**: Mini-batch approximation of gradient using subset of data

```python
grad_sum += compute_stoch_gradient(minibatch_y, minibatch_tx, w)
g = 1/B * grad_sum  # Average over B mini-batches
```

- **Stochastic approximation**: Uses batches to estimate full gradient
- **Computational efficiency**: $O(BD)$ vs $O(ND)$ per iteration
- **Noise injection**: Helps escape local minima

### 4. Ridge Regression (L2 Regularization)
**Mathematical Foundation**: Adds penalty term $\lambda\|\mathbf{w}\|^2$ to prevent overfitting

```python
w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(D), tx.T @ y)
```

- **Regularized normal equation**: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + 2N\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
- **Bias-variance tradeoff**: Parameter $\lambda$ controls regularization strength
- **Matrix conditioning**: Improves numerical stability

### 5. Logistic Regression
**Mathematical Foundation**: Probabilistic classification using sigmoid function

```python
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

pred = sigmoid(tx @ w)
loss = -np.mean(y * safe_log(pred) + (1 - y) * safe_log(1 - pred))
grad = tx.T @ (sigmoid(tx @ w) - y) / N
```

- **Sigmoid activation**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ maps to probabilities
- **Log-likelihood loss**: $L(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\sigma(\mathbf{x}_i^T\mathbf{w})) + (1-y_i)\log(1-\sigma(\mathbf{x}_i^T\mathbf{w}))]$
- **Gradient**: $\nabla L(\mathbf{w}) = \frac{1}{N}\mathbf{X}^T(\sigma(\mathbf{X}\mathbf{w}) - \mathbf{y})$

### 6. Regularized Logistic Regression
**Mathematical Foundation**: Adds L2 penalty to logistic regression

```python
grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
```

- **Regularized gradient**: $\nabla L_{reg}(\mathbf{w}) = \nabla L(\mathbf{w}) + 2\lambda\mathbf{w}$
- **Prevents overfitting**: Shrinks weights toward zero

## Advanced Mathematical Techniques

### Polynomial Feature Expansion
**Mathematical Foundation**: Creates non-linear features $\phi(\mathbf{x}) = [1, x, x^2, ..., x^d]$

```python
phi[n] = [x[n]**j for j in range(degree+1)]
```

- **Basis functions**: Transforms linear model to capture non-linear relationships
- **Feature mapping**: $\mathbf{x} \mapsto \phi(\mathbf{x})$ increases model capacity

### Cross-Validation for Model Selection
**Mathematical Foundation**: K-fold partitioning for unbiased performance estimation

```python
k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
```

- **Data partitioning**: Splits data into K folds for training/validation
- **Hyperparameter optimization**: Selects optimal $\lambda$ via grid search
- **Performance estimation**: Averages across folds for robust evaluation

### Principal Component Analysis (PCA)
**Mathematical Foundation**: Eigendecomposition for dimensionality reduction

```python
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
tx_pca = tx @ top_eigenvectors
```

- **Covariance matrix**: $\mathbf{C} = \frac{1}{N-1}\mathbf{X}^T\mathbf{X}$
- **Eigendecomposition**: Finds principal directions of variance
- **Dimensionality reduction**: Projects to lower-dimensional subspace

## Numerical Stability Considerations

- **Matrix conditioning**: Ridge regression improves condition number
- **Numerical clipping**: Prevents overflow in sigmoid and log functions
- **Convergence criteria**: Multiple stopping conditions for iterative methods
- **Safe logarithm**: Clamps values to prevent $\log(0)$

## Usage

```bash
python run.py  # Runs ridge regression pipeline with preprocessing
```

The implementation demonstrates mathematical rigor in ML algorithm development, with attention to numerical stability and computational efficiency.
