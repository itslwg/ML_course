# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # Do augmentation
    X_augmented = np.hstack([np.ones((x.shape[0], 1)), x[np.newaxis].T])

    if degree != 1:
        for j in np.arange(2, degree + 1):
            X_augmented = np.hstack([X_augmented, np.power(x, j)[np.newaxis].T])

    return X_augmented
