"""
Base class for implementing matrix factorization using alternating LS

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""
from functools import partial
import numpy as np
import scipy as sp
import scipy.sparse as ss
from tqdm import trange
from .utils import get_interaction_weights
# pylint: disable=invalid-name


class MF_ALS:
    """
    Alternating Least Square solution to Matrix Factorization
    """

    def __init__(
        self,
        num_features: int,
        reg_lambda: float,
        num_iters: int,
        initial_std: float,
        seed: int | None = None,
        verbose: bool = True
    ):
        self.reg_lambda = reg_lambda
        self.num_features = num_features
        self.num_iters = num_iters
        self.seed = seed
        self.initial_std = initial_std
        self.user_mat = None
        self.item_mat = None
        self.verbose = verbose
        self.reg_mat = np.eye(self.num_features) * self.reg_lambda

    def fit(self, train_mat):
        """
        Fit function
        """
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        rstate = np.random.RandomState(seed=self.seed)
        _, num_items = train_mat.shape
        self.item_mat = rstate.normal(
            loc=0.,
            scale=self.initial_std,
            size=(num_items, self.num_features)
        )
        loop = range(self.num_iters)
        if self.verbose:
            loop = trange(self.num_iters, desc=self.__class__.__name__)
        for _ in loop:
            self.user_mat = self.update(train_mat, self.item_mat)
            self.item_mat = self.update(train_mat.T, self.user_mat)

    def update(self, data_mat, latent_mat):
        """ALS update step"""
        inv_mat = np.linalg.inv(latent_mat.T.dot(latent_mat) + self.reg_mat)
        return data_mat.dot(latent_mat).dot(inv_mat)


class MF_WALS:
    """
    Weighted Alternating Least Square solution to Matrix Factorization
    """

    def __init__(
        self,
        num_features: int,
        reg_lambda: float,
        num_iters: int,
        initial_std: float,
        weighting_strategy: str = 'uniform',
        seed: int | None = None,
        verbose: bool = True
    ):
        self.reg_lambda = reg_lambda
        self.num_features = num_features
        self.num_iters = num_iters
        self.weighting_strategy = weighting_strategy
        self.seed = seed
        self.initial_std = initial_std
        self.user_mat = None
        self.item_mat = None
        self.weight_mat = None
        self.verbose = verbose
        self.reg_mat = np.eye(self.num_features) * self.reg_lambda

    def fit(self, train_mat):
        """
        Fit function
        """
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        rstate = np.random.RandomState(seed=self.seed)
        num_users, num_items = train_mat.shape
        self.item_mat = rstate.normal(
            loc=0.,
            scale=self.initial_std,
            size=(num_items, self.num_features)
        ).astype('float32')
        self.user_mat = np.zeros((num_users, self.num_features),
                                 dtype='float32')
        self.weight_mat = get_interaction_weights(
            train_mat=train_mat,
            strategy=self.weighting_strategy
        ).astype('float64')

        # create weight tensors
        item_wgt_tensor = np.zeros([num_users, num_items, num_items],
                                   dtype='float64')
        item_index = np.arange(num_items)
        item_wgt_tensor[:, item_index, item_index] = self.weight_mat
        user_wgt_tensor = np.zeros([num_items, num_users, num_users],
                                   dtype='float64')
        user_index = np.arange(num_users)
        user_wgt_tensor[:, user_index, user_index] = self.weight_mat.T

        loop = range(self.num_iters)
        if self.verbose:
            loop = trange(self.num_iters, desc=self.__class__.__name__)
        for _ in loop:
            # print(j, end="-", flush=True)
            self.user_mat = self.update(
                train_mat, self.item_mat, item_wgt_tensor)
            self.item_mat = self.update(
                train_mat.T, self.user_mat, user_wgt_tensor)

    def update(self, data_mat, latent_mat, wgt_tensor):
        """WALS update step"""
        Tmat = latent_mat.T @ wgt_tensor @ latent_mat
        Tmat = np.linalg.inv(Tmat + self.reg_mat)
        Tmat = np.expand_dims(data_mat @ latent_mat, 1) @ Tmat
        return Tmat[:, 0, :]

    # def update_vec(self, data_vec, latent_mat, wgts):
    #     """WALS update step"""
    #     Wtilde = np.diag(wgts)
    #     inv_mat = latent_mat.T.dot(Wtilde).dot(latent_mat) + self.reg_mat
    #     inv_mat = sp.linalg.inv(
    #         inv_mat, check_finite=False, overwrite_a=True)
    #     return data_vec.dot(Wtilde).dot(latent_mat).dot(inv_mat)

  # item_mat.T.dot(diag_wgt_mat.reshape()).dot(item_mat)

        # sp.linalg.inv(inv_mat, check_finite=False, overwrite_a=True)

        # user_tuple = [(train_mat[i, :], self.weight_mat[i, :])
        #               for i in range(num_users)]
        # item_tuple = [(train_mat[:, i].T, self.weight_mat[:, i])
        #               for i in range(num_items)]

# # try 1
        # for i, (train_vec, weight_vec) in enumerate(user_tuple):
        #     self.user_mat[i, :] = self.update(
        #         train_vec,
        #         self.item_mat,
        #         weight_vec
        #     )
        # for i, (train_vec, weight_vec) in enumerate(item_tuple):
        #     self.item_mat[i, :] = self.update(
        #         train_vec,
        #         self.user_mat,
        #         weight_vec
        #     )
        # try 2
        # for i in range(num_users):
        #     self.user_mat[i, :] = self.update(
        #         train_mat[i, :],
        #         self.item_mat,
        #         self.weight_mat[i, :]
        #     )
        # for i in range(num_items):
        #     self.item_mat[i, :] = self.update(
        #         train_mat[:, i].T,
        #         self.user_mat,
        #         self.weight_mat[:, i]
        #     )

        # Parrallel is taking more time than serial!!
        # with mp.Pool(processes=self.ncores) as p:
        #     results = p.map(lambda ituple: self.update(
        #         ituple[0],
        #         self.item_mat,
        #         ituple[1]
        #     ), user_tuple)
        #     #print(list(results), flush=True)
        #     for i, ix in enumerate(results):
        #         self.user_mat[i, :] = ix
        #     #self.user_mat = np.squeeze(np.asarray(list(results)), axis=1)

        # with mp.Pool(processes=self.ncores) as p:
        #     results = p.map(lambda ituple: self.update(
        #         ituple[0],
        #         self.user_mat,
        #         ituple[1]
        #     ), item_tuple)
        #     for i, ix in enumerate(results):
        #         self.item_mat[i, :] = ix
        # self.item_mat = np.squeeze(np.asarray(list(results)), axis=1)
        # print(np.squeeze(np.asarray(results), axis=1).shape)
