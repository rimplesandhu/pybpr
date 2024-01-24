"""
Class for sparse svd

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""
import numpy as np
from scipy.sparse import linalg
# pylint: disable=invalid-name


class MF_SVD:
    """
    Sparse SVD solution to Matrix Factorization
    """

    def __init__(
        self,
        num_features: int,
        num_iters: int,
        seed: int | None = None,
    ):

        self.num_features = num_features
        self.num_iters = num_iters
        self.seed = seed
        self.user_mat = None
        self.item_mat = None

    def fit(self, training_mat):
        """
        Fit function
        """
        training_mat = training_mat.astype(float)
        umat, smat, vhmat = linalg.svds(
            A=training_mat,
            k=self.num_features,
            maxiter=self.num_iters
        )
        self.user_mat = umat.dot(np.power(np.diag(smat), 0.5))
        self.item_mat = vhmat.T.dot(np.power(np.diag(smat), 0.5))
