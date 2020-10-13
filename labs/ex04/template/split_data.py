# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""

import numpy as np

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # setup indices
    n_train_samples = np.int(np.floor(x.shape[0] * ratio))
    train_indices = np.random.randint(0, x.shape[0], n_train_samples)
    test_indices = np.in1d(np.arange(x.shape[0]), train_indices, invert=True)
    # Split the data
    X_train, y_train = x[train_indices], y[train_indices]
    X_test, y_test = x[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test
