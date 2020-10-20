# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T, y))
    X = y[np.newaxis].T - np.dot(tx, w[np.newaxis].T)
    rmse = np.sqrt(1/tx.shape[0] * np.dot(X.T, X))

    return rmse, w
