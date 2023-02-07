"""Base class for defining Collaboative filtering"""
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from itertools import islice
# pylint: disable=invalid-name


class CFBase:
    """
    Base class for collaborative filtering set up
    """

    def __init__(
        self,
        users: List[int],
        items: List[int],
        name: str = 'Interactions'
    ):
        assert len(users) == len(items), 'User count not equal to item count'
        self.name = name
        self.df = pd.DataFrame({
            'User': users,
            'Item': items
        })
        for col in ['User', 'Item']:
            self.df[col] = self.df[col].astype('category')

        agg_flag = {'NumInteractions': ('Item', 'count')}
        self.df = self.df.merge(
            self.df.groupby('User').agg(**agg_flag).reset_index(),
            on='User'
        )
        self.R = csr_matrix(
            (
                np.ones((self.df.shape[0],)),
                (self.df['User'].cat.codes, self.df['Item'].cat.codes)
            ),
            dtype=np.int8
        )
        self.R_test = csr_matrix(self.R.shape, dtype=np.int8)
        self.R_train = csr_matrix(self.R.shape, dtype=np.int8)
        # self.R_test = csr_matrix(self.R.shape, dtype=np.int8)
        # self.R_train = csr_matrix(self.R.shape, dtype=np.int8)

    def generate_train_test(
        self,
        user_test_ratio: float = 0.2,
        min_item_interactions: int = 1
    ):
        """Split the R matrix into train and test"""
        while np.min(np.sum(self.R_train, axis=0)) < min_item_interactions:
            rstate = np.random.RandomState()
            self.R_test = dok_matrix(self.R.shape, dtype=np.int8)
            self.R_train = self.R.copy().todok()
            for ith_user in range(self.R.shape[0]):
                items_ith_user = self.R[ith_user].indices
                test_size = int(np.ceil(user_test_ratio * items_ith_user.size))
                ith_items = rstate.choice(
                    items_ith_user,
                    size=test_size,
                    replace=False
                )
                self.R_test[ith_user, ith_items] = 1
                self.R_train[ith_user, ith_items] = 0
            # print(self.R_train.toarray())
        self.R_test = self.R_test.tocsr()
        self.R_train = self.R_train.tocsr()

    # def get_top_items_for_this_user(
    #     self,
    #     R_est,
    #     user: int,
    #     exclude_training: bool = True
    # ):
    #     """Returns top products for this user"""
    #     user_pred =
    #     items_for_this_user = R_est[user, :]

    def get_top_items_for_this_user(
        self,
        iuser: int,
        userU,
        itemV,
        num_items: int
    ):
        """Returns top products for this user"""
        user_pred = userU[iuser].dot(itemV.T)
        liked = set(self.R_train[iuser].indices)
        top_inds = np.argsort(user_pred)[::-1]
        top_n = islice([ix for ix in top_inds if ix not in liked], num_items)
        return list(top_n)

    def get_similar(
        self,
        in_mat,
        for_this_inds: int,
        count: int = 5
    ):
        """
        Get similar pairs, items or users
        """
        # cosine distance is proportional to normalized euclidean distance,
        # thus we normalize the item vectors and use euclidean metric so
        # we can use the more efficient kd-tree for nearest neighbor search;
        # also the item will always to nearest to itself, so we add 1 to
        # get an additional nearest item and remove itself at the end
        normed_factors = normalize(in_mat)
        knn = NearestNeighbors(n_neighbors=count + 1, metric='euclidean')
        knn.fit(normed_factors)
        normed_factors = np.atleast_2d(normed_factors[for_this_inds])
        _, inds = knn.kneighbors(normed_factors)
        similar_inds = list(np.squeeze(inds.astype(np.uint32)))
        similar_inds = [ix for ix in similar_inds if ix != for_this_inds]
        return similar_inds

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f'{self.__class__.__name__}: {self.name}:\n'
        out_str += f'  Number of users = {self.R.shape[0]}\n'
        out_str += f'  Number of items = {self.R.shape[1]}\n'
        out_str += f'  Number of interactions = {self.R.nnz}\n'
        return out_str
