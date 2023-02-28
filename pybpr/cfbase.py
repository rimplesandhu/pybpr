"""Base class for defining Collaboative filtering"""
from typing import List
from itertools import islice
from functools import partial
import pathos.multiprocessing as mp
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from .utils import compute_ndcg

# import swifter
# pylint: disable=invalid-name


class UserItemInteractions:
    """
    Base class for collaborative filtering set up
    """

    def __init__(
        self,
        users: List[int],
        items: List[int],
        timestamps: List[int] | None = None,
        min_num_rating_per_user: int = 2,
        min_num_rating_per_item: int = 2,
        name: str = 'Interactions',
        num_cores: int = 8
    ):
        assert len(users) == len(items), 'User count not equal to item count'
        self.name = name
        self.ncores = num_cores
        user_col = 'UserID'
        item_col = 'ItemID'
        user_index_col = 'UserIndex'
        item_index_col = 'ItemIndex'
        num_ratings_col = 'NumRatings'
        self.df = pd.DataFrame({
            user_col: users,
            item_col: items
        })
        if timestamps is not None:
            self.df['Timestamp'] = timestamps

        # create user and item dframes
        agg_flag = {num_ratings_col: (item_col, 'count')}
        self.df_user = self.df.groupby(user_col).agg(**agg_flag).reset_index()
        agg_flag = {num_ratings_col: (user_col, 'count')}
        self.df_item = self.df.groupby(item_col).agg(**agg_flag).reset_index()

        # trim the interaction data based on min user item conditions
        ubool = (self.df_user[num_ratings_col] >= min_num_rating_per_user)
        selected_users = self.df_user.loc[ubool, user_col]
        ibool = (self.df_item[num_ratings_col] >= min_num_rating_per_item)
        selected_items = self.df_item.loc[ibool, item_col]
        user_bool = (self.df[user_col].isin(selected_users))
        item_bool = (self.df[item_col].isin(selected_items))
        self.df = self.df[item_bool & user_bool].reset_index(drop=True)
        user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
        self.df_user = self.df_user[user_bool]
        item_bool = self.df_item[item_col].isin(self.df[item_col].unique())
        self.df_item = self.df_item[item_bool]

        # slice the user and item dataframes
        iterator = zip([self.df_item, self.df_user], [item_col, user_col],
                       [item_index_col, user_index_col])
        for idf, id_colname, idx_colname in iterator:
            # idf = idf[idf[id_colname].isin(
            #     self.df[id_colname].unique())].copy()
            idf.sort_values(by=num_ratings_col, ascending=False, inplace=True)
            idf.reset_index(drop=True, inplace=True)
            idf.reset_index(drop=False, inplace=True)
            idf.set_index(id_colname, drop=True, inplace=True)
            self.df[idx_colname] = idf.loc[self.df[id_colname]]['index'].values
            idf.rename(columns={'index': idx_colname}, inplace=True)
            idf.reset_index(drop=False, inplace=True)
            idf.set_index(idx_colname, drop=True, inplace=True)
        self.num_users = self.df[user_col].nunique()
        self.num_items = self.df[item_col].nunique()
        # self.df_user[user_col] = self.df_user[user_col].astype('category')
        # self.df_item[item_col] = self.df_item[item_col].astype('category')

        self.R = csr_matrix(
            (np.ones((self.df.shape[0],)),
             (self.df['UserIndex'], self.df['ItemIndex'])),
            dtype=np.int8
        )
        self.R_test = csr_matrix(self.R.shape, dtype=np.int8)
        self.R_train = csr_matrix(self.R.shape, dtype=np.int8)
        # for col in [user_col, item_col, user_index_col, item_index_col]:
        #     self.df[col] = self.df[col].astype('category')

    @property
    def sparsity(self):
        """Get the sparsity"""
        return self.df.shape[0] * 100 / (self.num_users * self.num_items)

    def print_memory_usage(self):
        """Prints memory usage"""
        print(f'--- Memory usage for {self.name}:', flush=True)
        R_mb = np.around(self.R.data.nbytes / 1024 / 1024, 2)
        print(f'Sparse User-Item matrix = {R_mb} MB', flush=True)
        df_mb = np.around(self.df.memory_usage().sum() / 1024 / 1024, 2)
        print(f'User-Item dataframe df = {df_mb} MB', flush=True)
        df_mb = np.around(self.df_item.memory_usage().sum() / 1024 / 1024, 2)
        print(f'Item dataframe df_item = {df_mb} MB', flush=True)
        df_mb = np.around(self.df_user.memory_usage().sum() / 1024 / 1024, 2)
        print(f'Item dataframe df_user = {df_mb} MB', flush=True)
        print('---')

    def generate_train_test(
        self,
        user_test_ratio: float = 0.2,
        seed: int = 1234
    ):
        """Split the R matrix into train and test"""
        rstate = np.random.RandomState(seed=seed)
        self.df['Training'] = True
        index_names = ['ItemIndex', 'UserIndex']
        if list(self.df.index.names) != index_names:
            self.df.set_index(index_names, inplace=True)
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
            self.df.loc[(ith_items, ith_user), 'Training'] = False
        self.R_test = self.R_test.tocsr()
        self.R_train = self.R_train.tocsr()

    def get_top_items_for_this_user(
        self,
        user_idx: int,
        user_mat,
        item_mat,
        num_items: int,
        exclude_liked: bool = True
    ):
        """Returns top products for this user"""
        user_pred = user_mat[user_idx].dot(item_mat.T)
        top_inds = np.argsort(user_pred)[::-1]
        liked = set(self.R_train[user_idx].indices) if exclude_liked else set()
        top_n = islice([ix for ix in top_inds if ix not in liked], num_items)
        # top_val = islice([user_pred[ix]
        #                  for ix in top_inds if ix not in liked], num_items)
        return list(top_n)

    def get_ndcg_metric_user(
        self,
        user_idx: int,
        user_mat,
        item_mat,
        num_items: int,
        test: bool = True,
        truncate: bool = True
    ):
        """Computes NDCG score for this user"""
        data_mat = self.R_test if test else self.R_train
        test_inds = list(data_mat[user_idx].indices)
        num_items = min(len(test_inds), num_items) if truncate else num_items
        exclude_liked = True if test else False
        top_items = self.get_top_items_for_this_user(
            user_idx=user_idx,
            user_mat=user_mat,
            item_mat=item_mat,
            num_items=num_items,
            exclude_liked=exclude_liked
        )
        ndcg_score = compute_ndcg(
            ranked_item_idx=np.where(np.isin(top_items, test_inds))[0],
            K=num_items,
            wgt_fun=np.log2
        )
        return ndcg_score

    def get_ndcg_metric(
        self,
        user_mat,
        item_mat,
        num_items: int,
        test: bool = True,
        truncate: bool = True,
        ncores: int = 1
    ):
        """Averaged NDCG across all users"""
        pfunc = partial(
            self.get_ndcg_metric_user,
            user_mat=user_mat,
            item_mat=item_mat,
            num_items=num_items,
            test=test,
            truncate=truncate
        )
        ndcg_scores = []
        if ncores <= 1:
            for idx in list(range(self.num_users)):
                ndcg_scores.append(pfunc(idx))
        else:
            with mp.Pool(processes=self.ncores) as p:
                ndcg_scores = p.map(pfunc, list(range(self.num_users)))
        return np.mean(ndcg_scores)

    def get_similar(
        self,
        feature_mat,
        idx: int,
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
        normed_factors = normalize(feature_mat)
        knn = NearestNeighbors(n_neighbors=count + 1, metric='euclidean')
        knn.fit(normed_factors)
        normed_factors = np.atleast_2d(normed_factors[idx])
        _, inds = knn.kneighbors(normed_factors)
        similar_inds = list(np.squeeze(inds.astype(np.uint32)))
        similar_inds = [ix for ix in similar_inds if ix != idx]
        return similar_inds

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f'{self.__class__.__name__}: {self.name}:\n'
        out_str += f'  Number of users = {self.R.shape[0]}\n'
        out_str += f'  Number of items = {self.R.shape[1]}\n'
        out_str += f'  Number of interactions = {self.R.nnz}\n'
        return out_str


cf_basic = UserItemInteractions(
    users=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
           5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
    items=[1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4,
           6, 7, 8, 9, 10, 6, 7, 8, 9, 7, 8, 9, 10, 7, 8, 9]
)

# def get_idx(entry):
#     print(entry[item_col])
#     item_idx = pd.Index(
#         self.df_item[item_col]).get_loc(entry[item_col].values)
#     user_idx = pd.Index(
#         self.df_user[user_col]).get_loc(entry[user_col].values)
#     return item_idx, user_idx
# indices_tuple = self.df.swifter.apply(get_idx, axis=1)
# item_inds, user_inds = zip(*indices_tuple)
# user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
# self.df_user = self.df_user[user_bool]
# self.df_user.sort_values(by=num_ratings_col, ascending=False,
#                          inplace=True)
# self.df_user.reset_index(drop=True, inplace=True)
# item_bool = self.df_item[item_col].isin(self.df[item_col].unique())
# self.df_item = self.df_item[item_bool]
# self.df_item.sort_values(by=num_ratings_col, ascending=False,
#                          inplace=True)
# self.df_item.reset_index(drop=True, inplace=True)

# number of user and items

# # construct the user - item matrix
# self.df_user.reset_index(drop=False, inplace=True)
# self.df_user.set_index(user_col, drop=True, inplace=True)
# self.df['UserIdx'] = self.df_user.loc[self.df[user_col]]['index'].values
# self.df_user.drop(columns=('index'))

# self.df_item.reset_index(drop=False, inplace=True)
# self.df_item.set_index(item_col, drop=True, inplace=True)
# self.df['ItemIdx'] = self.df_item.loc[self.df[item_col]]['index'].values
# self.df_item.drop(columns=('index'))
# u_inds = self.df_user[user_col].cat.codes.values
# i_inds = self.df_item[item_col].cat.codes.values
# self.df_user['NumRatings_Train'] = self.R_train.sum(axis=1)[u_inds, 0]
# self.df_user['NumRatings_Test'] = self.R_test.sum(axis=1)[u_inds, 0]
# self.df_item['NumRatings_Train'] = self.R_train.sum(axis=0)[0, i_inds]
# self.df_item['NumRatings_Test'] = self.R_test.sum(axis=0)[0, i_inds]

# def get_top_items_for_this_user(
#     self,
#     R_est,
#     user: int,
#     exclude_training: bool = True
# ):
#     """Returns top products for this user"""
#     user_pred =
#     items_for_this_user = R_est[user, :]
