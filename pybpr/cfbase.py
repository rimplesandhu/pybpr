"""Base class for defining Collaboative filtering"""
from typing import List
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

#pylint: disable=invalid-name


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
        self.R_train = self.R.copy()
        #self.R_test = csr_matrix(self.R.shape, dtype=np.int8)
        #self.R_train = csr_matrix(self.R.shape, dtype=np.int8)

    def generate_train_test(
        self,
        test_ratio: float = 0.2,
        rng_seed: int = 1234
    ):
        """Split the R matrix into train and test"""
        rstate = np.random.RandomState(seed=rng_seed)
        self.R_test = self.R_test.todok()
        self.R_train = self.R_train.todok()
        for ith_user in range(self.R.shape[0]):
            items_for_ith_user = self.R[ith_user].indices
            test_size = int(np.ceil(test_ratio * items_for_ith_user.size))
            ith_items = rstate.choice(
                items_for_ith_user,
                size=test_size,
                replace=False
            )
            self.R_test[ith_user, ith_items] = 1
            self.R_train[ith_user, ith_items] = 0
        self.R_test = self.R_test.tocsr()
        self.R_train = self.R_train.tocsr()

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f'{self.__class__.__name__}: {self.name}:\n'
        out_str += f'  Number of users = {self.R.shape[0]}\n'
        out_str += f'  Number of items = {self.R.shape[1]}\n'
        out_str += f'  Number of interactions = {self.R.nnz}\n'
        return out_str
