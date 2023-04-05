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
# pylint: disable=invalid-name
#


class BPR:
    """
    Bayesian Personalized Ranking (BPR)
    """

    def __init__(
        self,
        num_features: int,
        num_iters: int,
        reg_lambda: float,
        learning_rate: float,
        batch_size: int,
        initial_std: float,
        seed: int | None = None,
        verbose: bool = True
    ):
        self.num_features = int(num_features)
        self.num_iters = int(num_iters)
        self.batch_size = int(batch_size)
        self.user_mat = None
        self.item_mat = None
        self.initial_std = initial_std
        self.user_item_mat = None
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.seed = seed
        self.ndcg_metric = []
        # to avoid re-computation at predict
        # self._prediction = None

    def fit(self, train_mat, ndcg_func=None):
        """
        Fit function
        """
        num_users, num_items = train_mat.shape
        indptr = train_mat.indptr
        indices = train_mat.indices
        if num_users < self.batch_size:
            batch_size = num_users
            print('WARNING: Batch size is greater than number of users')
            sys.stderr.write(f'setting batch size to {num_users}\n')
        else:
            batch_size = self.batch_size
        batch_iters = num_users // batch_size

        # initialize random
        rstate = np.random.RandomState(self.seed)
        self.user_mat = rstate.normal(
            loc=0.,
            scale=self.initial_std,
            size=(num_users, self.num_features)
        )
        self.item_mat = rstate.normal(
            loc=0.,
            scale=self.initial_std,
            size=(num_items, self.num_features)
        )

        # progress bar for training iteration if verbose is turned on
        self.ndcg_metric = []
        loop = range(self.num_iters)
        if self.verbose:
            loop = trange(self.num_iters, desc=self.__class__.__name__)
        for _ in loop:
            for _ in range(batch_iters):
                sampled_pos_items = np.zeros(batch_size, dtype=np.int)
                sampled_neg_items = np.zeros(batch_size, dtype=np.int)
                sampled_users = np.random.choice(
                    a=num_users,
                    size=batch_size,
                    replace=False
                )
                for idx, user in enumerate(sampled_users):
                    pos_items = indices[indptr[user]:indptr[user + 1]]
                    pos_item = np.random.choice(pos_items)
                    neg_item = np.random.choice(num_items)
                    while neg_item in pos_items:
                        neg_item = np.random.choice(num_items)
                    sampled_pos_items[idx] = pos_item
                    sampled_neg_items[idx] = neg_item
                self.update(
                    sampled_users,
                    sampled_pos_items,
                    sampled_neg_items
                )
            if ndcg_func is not None:
                ndcg_test = ndcg_func(
                    user_mat=self.user_mat,
                    item_mat=self.item_mat,
                    test=True
                )
                ndcg_train = ndcg_func(
                    user_mat=self.user_mat,
                    item_mat=self.item_mat,
                    test=False
                )
                self.ndcg_metric.append(
                    {'test': ndcg_test, 'train': ndcg_train}
                )

    # def _sample(self, batch_size, indices, indptr):
    #     """sample batches of random triplets u, i, j"""
    #     sampled_pos_items = np.zeros(batch_size, dtype=np.int)
    #     sampled_neg_items = np.zeros(batch_size, dtype=np.int)
    #     sampled_users = np.random.choice(
    #         self.num_users, size=batch_size, replace=False)

    #     for idx, user in enumerate(sampled_users):
    #         pos_items = indices[indptr[user]:indptr[user + 1]]
    #         pos_item = np.random.choice(pos_items)
    #         neg_item = np.random.choice(self.num_items)
    #         while neg_item in pos_items:
    #             neg_item = np.random.choice(self.num_items)
    #         sampled_pos_items[idx] = pos_item
    #         sampled_neg_items[idx] = neg_item

    #     return sampled_users, sampled_pos_items, sampled_neg_items

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
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg_lambda * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg_lambda * item_i
        grad_j = sigmoid_tiled * user_u + self.reg_lambda * item_j
        self.user_mat[u] -= self.learning_rate * grad_u
        self.item_mat[i] -= self.learning_rate * grad_i
        self.item_mat[j] -= self.learning_rate * grad_j
