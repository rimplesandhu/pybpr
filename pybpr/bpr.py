"""Base class for implementing matrix factorization"""
from typing import List
import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import inv
from sklearn.metrics import mean_squared_error
from tqdm import trange
from .utils import compute_mse
#pylint: disable=invalid-name
#


class BPR:
    """
    Bayesian Personalized Ranking (BPR) for implicit feedback data

    Parameters
    ----------
    learning_rate : float, default 0.01
        learning rate for gradient descent

    n_factors : int, default 20
        Number/dimension of user and item latent factors

    n_iters : int, default 15
        Number of iterations to train the algorithm

    batch_size : int, default 1000
        batch size for batch gradient descent, the original paper
        uses stochastic gradient descent (i.e., batch size of 1),
        but this can make the training unstable (very sensitive to
        learning rate)

    reg : int, default 0.01
        Regularization term for the user and item latent factors

    seed : int, default 1234
        Seed for the randomly initialized user, item latent factors

    verbose : bool, default True
        Whether to print progress bar while training

    Attributes
    ----------
    user_factors : 2d ndarray, shape [self.num_users, n_factors]
        User latent factors learnt

    item_factors : 2d ndarray, shape [self.num_items, n_factors]
        Item latent factors learnt

    References
    ----------
    S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme 
    Bayesian Personalized Ranking from Implicit Feedback
    - https://arxiv.org/abs/1205.2618
    """

    def __init__(
        self,
        num_features: int,
        num_users: int,
        num_items: int,
        reg_lambda: float,
        learning_rate: float,
        verbose: bool = False
    ):
        self.num_features = num_features
        self.num_users = num_users
        self.num_items = num_items
        self.user_mat = None
        self.item_mat = None
        self.user_item_mat = None
        self.user_item_mateg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.test_mse = []
        self.train_mse = []

        # self.user_item_mateg = reg
        # self.verbose = verbose
        # self.n_iters = n_iters
        # self.num_features
        #  = n_factors
        # self.batch_size = batch_size
        # self.learning_rate = learning_rate

        # to avoid re-computation at predict
        # self._prediction = None

    def fit(
        self,
        R_train,
        R_test=None,
        batch_size: int = 2,
        num_iters: int = 10,
        store_mse: bool = True,
        seed: int = 1234
    ):
        """
        Parameters
        ----------
        R_train : scipy sparse csr_matrix, shape [self.num_users, self.num_items]
            sparse matrix of user-item interactions
        """
        self.test_mse = []
        self.train_mse = []
        indptr = R_train.indptr
        indices = R_train.indices
        if self.num_users < batch_size:
            batch_size = self.num_users
            print('WARNING: Batch size is greater than number of users')
            sys.stderr.write(f'setting batch size to {self.num_users}\n')

        batch_iters = self.num_users // batch_size

        # initialize random weights
        rstate = np.random.RandomState(seed)
        self.user_mat = rstate.normal(
            size=(self.num_users, self.num_features))
        self.item_mat = rstate.normal(
            size=(self.num_items, self.num_features))

        # progress bar for training iteration if verbose is turned on
        loop = range(num_iters)
        if self.verbose:
            loop = trange(num_iters, desc=self.__class__.__name__)

        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(batch_size, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self.update(
                    sampled_users,
                    sampled_pos_items,
                    sampled_neg_items
                )
            if store_mse:
                self.user_item_mat = self.user_mat.dot(self.item_mat.T)
                self.train_mse.append(compute_mse(R_train, self.user_item_mat))
                if R_test is not None:
                    self.test_mse.append(compute_mse(
                        R_test, self.user_item_mat))
        return self

    def _sample(self, batch_size, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(batch_size, dtype=np.int)
        sampled_neg_items = np.zeros(batch_size, dtype=np.int)
        sampled_users = np.random.choice(
            self.num_users, size=batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(self.num_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(self.num_items)
            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items

    def update(self, u, i, j):
        """
        update according to the bootstrapped user u, 
        positive item i and negative item j
        """
        user_u = self.user_mat[u]
        item_i = self.item_mat[i]
        item_j = self.item_mat[j]

        # decompose the estimator, compute the difference between
        # the score of the positive items and negative items; a
        # naive implementation might look like the following:
        # r_ui = np.diag(user_u.dot(item_i.T))
        # r_uj = np.diag(user_u.dot(item_j.T))
        # r_uij = r_ui - r_uj

        # however, we can do better, so
        # for batch dot product, instead of doing the dot product
        # then only extract the diagonal element (which is the value
        # of that current batch), we perform a hadamard product,
        # i.e. matrix element-wise product then do a sum along the column will
        # be more efficient since it's less operations
        # http://people.revoledu.com/kardi/tutorial/LinearAlgebra/HadamardProduct.html
        # r_ui = np.sum(user_u * item_i, axis = 1)
        #
        # then we can achieve another speedup by doing the difference
        # on the positive and negative item up front instead of computing
        # r_ui and r_uj separately, these two idea will speed up the operations
        # from 1:14 down to 0.36
        r_uij = np.sum(user_u * (item_i - item_j), axis=1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))

        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.num_features, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + \
            self.user_item_mateg_lambda * user_u
        grad_i = sigmoid_tiled * -user_u + self.user_item_mateg_lambda * item_i
        grad_j = sigmoid_tiled * user_u + self.user_item_mateg_lambda * item_j
        self.user_mat[u] -= self.learning_rate * grad_u
        self.item_mat[i] -= self.learning_rate * grad_i
        self.item_mat[j] -= self.learning_rate * grad_j
