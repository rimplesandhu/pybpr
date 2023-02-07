"""Utilities"""


import numpy as np
from sklearn.metrics import mean_squared_error


def compute_mse(y_true, y_pred):
    """ignore zero terms prior to comparing the mse"""
    mask = np.nonzero(y_true)
    assert mask[0].shape[0] > 0, 'Truth matrix empty'
    mse = mean_squared_error(np.array(y_true[mask]).ravel(), y_pred[mask])
    return mse
