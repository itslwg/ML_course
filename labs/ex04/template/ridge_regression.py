# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def compute_ridge_loss(y, tx, w, lambda_):
    """Compue RMSE loss in matrix form."""
    e = y[np.newaxis].T - np.dot(tx, w[np.newaxis].T)
    rmse = np.sqrt(1/(tx.shape[0]) * np.dot(e.T, e) + lambda_ * np.dot(w, w.T))

    return rmse.ravel()


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    a = (tx.T @ tx) + lambda_ * 2 * tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    rmse = compute_ridge_loss(y, tx, w, lambda_)

    return rmse, w

