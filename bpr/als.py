"""Base class for implementing matrix factorization"""
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import inv
from sklearn.metrics import mean_squared_error
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
        num_iters: int,
        num_features: int,
        num_users: int,
        num_items: int,
        reg_lambda: float
    ):
        self.reg_lambda = reg_lambda
        self.num_iters = num_iters
        self.num_features = num_features
        istr = 'ALS: # of features are more than min(num_items, num_users)!'
        assert num_features < min(num_users, num_items), istr
        self.num_users = num_users
        self.num_items = num_items
        self.U = np.random.random((num_users, self.num_features))
        self.V = np.random.random((num_items, self.num_features))
        self.test_mse_record = []
        self.train_mse_record = []

    def fit(
        self,
        R_train,
        R_test=None
    ):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        for _ in range(self.num_iters):
            self.U = self.update(R_train, self.V)
            self.V = self.update(R_train.T, self.U)
            R_pred = self.predict()
            # print(R_pred.shape)
            self.train_mse_record.append(self.compute_mse(R_train, R_pred))
            if R_test is not None:
                self.test_mse_record.append(self.compute_mse(R_test, R_pred))

    def update(self, R_mat, Mfixed):
        """ALS update step"""
        Amat = Mfixed.T.dot(Mfixed)
        Amat += np.eye(self.num_features) * self.reg_lambda
        # print(Amat.shape)
        return R_mat.dot(Mfixed).dot(np.linalg.inv(Amat))

    def predict(self):
        """Predicts user-item interaction matrix"""
        return self.U.dot(self.V.T)

    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(np.array(y_true[mask]).ravel(), y_pred[mask])
        return mse
