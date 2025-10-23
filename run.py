import numpy as np

from helpers import *
from plots import *
from cross_validation import *
from build_polynomial import *
from validation import *
from preproccessing import *
from implementations import *

from gradient_descent import *
from stochastic_gradient_descent import *
from least_squares import *
from ridge_regression import *
from logistic_regression import *


def main():

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/", sub_sample=False)

    # Define constants
    names = ["Least Squares", "Gradient Descent", "Stochastic Gradient Descent", "Ridge Regression", "Logistic Regression", "Regularized Logistic Regression"]
    max_iters = 1000
    seed = 123

    # Define hyperparameters
    gamma = 0.1
    lambda_ridge = 0.01
    lambda_logistic = 0.001

    # Preprocess data
    tx_train, tx_validate, tx_test, y, y_validate = preprocess_data(
                                        x_train[:, 9:],
                                        x_test[:, 9:],
                                        y_train,
                                        min_unique = 7,
                                        var_tol = 0.0001,
                                        corr_tol = 0.999,
                                        empty_tol = 0.90,
                                        num_components = 300,
                                        validation_size = 0.10,
                                        ratio = 0.10,
                                        seed = 1,
                                        pca = False,
                                        hon = True
    )

    # Calculate classification
    w_rr, mse_rr = ridge_regression(y, tx_train, lambda_ridge)

    # Calculate classification
    cutoffs = np.linspace(0.10, 0.99, 51)
    cutoff_rr, f1_scores_rr = cross_validation_cutoffs(y_validate, tx_validate, w_rr, predict_class, cutoffs, plot=False) # 0.66
    pred = np.where(tx_test @ w_rr > cutoff_rr, 1, -1)
    print(len(pred[pred == 1])/len(pred))
    create_csv_submission(test_ids, pred, "data/sample_submission.csv")

main()
