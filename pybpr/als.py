"""Base class for implementing matrix factorization"""
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import inv
from .utils import compute_mse
#pylint: disable=invalid-name


class ALS:
    """
    Alternating Least Square

    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix

    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm

    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank

    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(
        self,
        num_features: int,
        num_users: int,
        num_items: int,
        reg_lambda: float
    ):
        self.reg_lambda = reg_lambda
        self.num_features = num_features
        self.num_users = num_users
        self.num_items = num_items
        istr = 'ALS: # of features are more than min(num_items, num_users)!'
        assert num_features < min(num_users, num_items), istr
        self.user_mat = None
        self.item_mat = None
        self.user_item_mat = None
        self.test_mse = []
        self.train_mse = []

    def fit(
        self,
        R_train,
        R_test=None,
        num_iters: int = 10,
        store_mse: bool = True,
        seed: int | None = None
    ):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.test_mse = []
        self.train_mse = []
        np.random.seed(seed=seed)
        self.user_mat = np.random.random(
            (self.num_users, self.num_features))
        self.item_mat = np.random.random((self.num_items, self.num_features))
        for _ in range(num_iters):
            self.user_mat = self.update(R_train, self.item_mat)
            self.item_mat = self.update(R_train.T, self.user_mat)
            self.user_item_mat = self.user_mat.dot(self.item_mat.T)
            if store_mse:
                self.train_mse.append(compute_mse(R_train, self.user_item_mat))
                if R_test is not None:
                    self.test_mse.append(compute_mse(
                        R_test, self.user_item_mat))

    def update(self, R_mat, Mfixed):
        """ALS update step"""
        Amat = Mfixed.T.dot(Mfixed)
        Amat += np.eye(self.num_features) * self.reg_lambda
        return R_mat.dot(Mfixed).dot(np.linalg.inv(Amat))
