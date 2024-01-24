"""
Base class for defining User-Item interaction data

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, dok_matrix, spmatrix

# pylint: disable=invalid-name


class UserItemInteractions:
    """
    Base class for collaborative filtering set up
    """

    def __init__(
        self,
        items_index: list[int],
        users_index: list[int],
        num_items: int = None,
        num_users: int = None,
        name: str = 'Sample',
    ):
        # Initiate
        items_index = np.asarray(items_index, dtype=np.int64)
        users_index = np.asarray(users_index, dtype=np.int64)
        self.name = name

        # check compatibility of user inputs
        if items_index.size != users_index.size:
            raise ValueError('Length mismatch b/w items_index and users_index')

        # deal with num_items
        if num_items is not None:
            if max(items_index) >= num_items:
                raise ValueError('num_items incompatible with items_index!')
        else:
            num_items = np.amax(items_index) + 1

        # deal with num_users
        if num_users is not None:
            if max(users_index) >= num_users:
                raise ValueError('num_items incompatible with items_index!')
        else:
            num_users = np.amax(users_index) + 1

        # create an interaction matrix sparsely filled with ones
        self._mat = csr_matrix(
            # (np.ones((len(items_index),)), (users_index, items_index)),
            (np.repeat(True, len(items_index)), (users_index, items_index)),
            shape=(num_users, num_items),
            dtype=bool
        )

        # # check no empty rows or columns
        # assert np.amin(self._mat.sum(axis=0)) > 0, 'need atleast 1 entry!'
        # assert np.amin(self._mat.sum(axis=1)) > 0, 'need atleast 1 entry!'

        # create training and test versions of sparse int matrix
        self._mat_test = csr_matrix(self.mat.shape, dtype=np.int8)
        self._mat_train = self.mat.copy()
        print(self.__str__())

    def generate_train_test(
        self,
        user_test_ratio: float = 0.2,
        seed: int = 1234
    ):
        """Split the UI matrix into train and test"""
        print('Generating train-test split..', end="")
        assert user_test_ratio <= 0.5, 'user_test_ratio should be in [0,0.5]'
        assert user_test_ratio >= 0.0, 'user_test_ratio should be in [0,0.5]'

        if user_test_ratio < 1e-3:
            print('Warning: Test matrix is set as empty/all-zeros', flush=True)
            self._mat_test = csr_matrix(self.mat.shape, dtype=bool)
            self._mat_train = self.mat.copy()
        else:
            # min number of item interactions required per user
            min_interactions = int(1/user_test_ratio) + 1

            # find those users that haave atleast min_interactions
            num_interactions = np.asarray(self.mat.sum(axis=1)).reshape(-1)
            valid_users = np.where(num_interactions > min_interactions)[0]

            # initiate test and train matrices
            np.random.seed(seed=seed)
            self._mat_test = dok_matrix(self.mat.shape, dtype=bool)
            self._mat_train = self.mat.copy().todok()

            # iterate over each user and split its interactions into test/train
            for ith_user in valid_users:
                items_ith_user = self._mat[ith_user].indices
                test_size = int(np.ceil(user_test_ratio * items_ith_user.size))
                ith_items = np.random.choice(
                    items_ith_user,
                    size=test_size,
                    replace=False
                )
                self._mat_test[ith_user, ith_items] = True
                self._mat_train[ith_user, ith_items] = False
            self._mat_test = self.mat_test.tocsr()
            self._mat_train = self.mat_train.tocsr()
            print('done', flush=True)

        # check if train+test=original UI mat
        if (self.mat_train + self.mat_test != self.mat).nnz != 0:
            raise RuntimeError('Issue with test/train split')

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f'\n----{self.__class__.__name__}--{self.name}\n'
        out_str += f'# of users (active/total): {self.num_users_active}/{
            self.num_users}\n'
        out_str += f'# of items (active/total): {self.num_items_active}/{
            self.num_items}\n'
        out_str += f'# of interactions: {self.mat.nnz}\n'
        out_str += f'Sparsity in the UI mat: {self.sparsity}\n'
        R_mb = np.around(self.mat.data.nbytes / 1024 / 1024, 2)
        out_str += f'Memory used by sparse UI mat: {R_mb} MB'
        return out_str

    @property
    def mat(self) -> spmatrix:
        """Number of users"""
        return self._mat

    @property
    def mat_test(self) -> spmatrix:
        """Number of users"""
        return self._mat_test

    @property
    def mat_train(self) -> spmatrix:
        """Number of users"""
        return self._mat_train

    @property
    def num_users(self) -> int:
        """Number of users"""
        return self.mat.shape[0]

    @property
    def num_items(self) -> int:
        """Number of users"""
        return self.mat.shape[1]

    @property
    def active_users(self) -> ndarray:
        """Index of users with atleast one interaction"""
        return np.where(np.asarray(self.mat.sum(axis=1)).reshape(-1) > 0)[0]

    @property
    def active_items(self) -> ndarray:
        """Index of items with atleast one interaction"""
        return np.where(np.asarray(self.mat.sum(axis=0)).reshape(-1) > 0)[0]

    @property
    def num_users_active(self) -> int:
        """Number of users"""
        return self.active_users.size

    @property
    def num_items_active(self) -> int:
        """Number of users"""
        return self.active_items.size

    @property
    def sparsity(self) -> float:
        """Get the sparsity"""
        outs = self.mat.nnz * 1 / (self.num_users * self.num_items)
        return np.around(outs, 6)


# Junk
    # def users_sorted_by_activity(self, count: int | None = None):
    #     """return the most -count- active users"""
    #     # get number of interactions for each user
    #     count_vector = np.asarray(self._mat.sum(axis=1)).reshape(-1)
    #     # get the index of users with most interactions at 0
    #     sorted_user_list = count_vector.argsort()[::-1]
    #     if count is not None:
    #         sorted_user_list = sorted_user_list[:count]
    #     return sorted_user_list

    # def get_ndcg_metric_user(
    #     self,
    #     user_idx: int,
    #     user_mat,
    #     item_mat,
    #     num_items: int,
    #     test: bool = True,
    #     truncate: bool = True
    # ):
    #     """Computes NDCG score for this user"""
    #     data_mat = self._mat_test if test else self._mat_train
    #     test_inds = list(data_mat[user_idx].indices)
    #     num_items = min(len(test_inds), num_items) if truncate else num_items
    #     exclude_liked = True if test else False
    #     top_items = self.get_top_items_for_this_user(
    #         user_idx=user_idx,
    #         user_mat=user_mat,
    #         item_mat=item_mat,
    #         num_items=num_items,
    #         exclude_liked=exclude_liked
    #     )
    #     ndcg_score = compute_ndcg(
    #         ranked_item_idx=np.where(np.isin(top_items, test_inds))[0],
    #         K=num_items,
    #         wgt_fun=np.log2
    #     )
    #     return ndcg_score

    # # def get_ndcg_metric(
    #     self,
    #     user_mat,
    #     item_mat,
    #     num_items: int,
    #     num_users: int = 100,
    #     test: bool = True,
    #     truncate: bool = False,
    #     ncores:int=1
    # ):
    #     """Averaged NDCG across all users"""
    #     pfunc = partial(
    #         self.get_ndcg_metric_user,
    #         user_mat=user_mat,
    #         item_mat=item_mat,
    #         num_items=num_items,
    #         test=test,
    #         truncate=truncate
    #     )
    #     ndcg_scores = []
    #     chosen_users = np.random.choice(
    #         self.num_users,
    #         size=num_users
    #     )
    #     if self.ncores <= 1:
    #         for idx in chosen_users:
    #             ndcg_scores.append(pfunc(idx))
    #     else:
    #         with mp.Pool(processes=self.ncores) as p:
    #             ndcg_scores = p.map(pfunc, chosen_users)
    #     return np.mean(ndcg_scores)

    # def get_similar(
    #     self,
    #     feature_mat,
    #     idx: int,
    #     count: int = 5
    # ):
    #     """
    #     Get similar pairs, items or users
    #     """
    #     # cosine distance is proportional to normalized euclidean distance,
    #     # thus we normalize the item vectors and use euclidean metric so
    #     # we can use the more efficient kd-tree for nearest neighbor search;
    #     # also the item will always to nearest to itself, so we add 1 to
    #     # get an additional nearest item and remove itself at the end
    #     normed_factors = normalize(feature_mat)
    #     knn = NearestNeighbors(n_neighbors=count + 1, metric='euclidean')
    #     knn.fit(normed_factors)
    #     normed_factors = np.atleast_2d(normed_factors[idx])
    #     _, inds = knn.kneighbors(normed_factors)
    #     similar_inds = list(np.squeeze(inds.astype(np.uint32)))
    #     similar_inds = [ix for ix in similar_inds if ix != idx]
    #     return similar_inds


# cf_8user_10item = UserItemInteractions(
#     users=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
#            5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
#     items=[1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4,
#            6, 7, 8, 9, 10, 6, 7, 8, 9, 7, 8, 9, 10, 7, 8, 9]
# )

# cf_4user_5item = UserItemInteractions(
#     users=[1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
#     items=[1, 2, 3, 5, 1, 2, 4, 2, 3, 3],
#     min_num_rating_per_user=0,
#     min_num_rating_per_item=0,
# )

# # def get_idx(entry):
# #     print(entry[self.item_col])
# #     item_idx = pd.Index(
# #         self.df_item[self.item_col]).get_loc(entry[self.item_col].values)
# #     user_idx = pd.Index(
# #         self.df_user[user_col]).get_loc(entry[user_col].values)
# #     return item_idx, user_idx
# # indices_tuple = self.df.swifter.apply(get_idx, axis=1)
# # item_inds, user_inds = zip(*indices_tuple)
# # user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
# # self.df_user = self.df_user[user_bool]
# # self.df_user.sort_values(by=num_ratings_col, ascending=False,
# #                          inplace=True)
# # self.df_user.reset_index(drop=True, inplace=True)
# # item_bool = self.df_item[self.item_col].isin(self.df[self.item_col].unique())
# # self.df_item = self.df_item[item_bool]
# # self.df_item.sort_values(by=num_ratings_col, ascending=False,
#                          inplace=True)
# self.df_item.reset_index(drop=True, inplace=True)

# number of user and items

# # construct the user - item matrix
# self.df_user.reset_index(drop=False, inplace=True)
# self.df_user.set_index(user_col, drop=True, inplace=True)
# self.df['UserIdx'] = self.df_user.loc[self.df[user_col]]['index'].values
# self.df_user.drop(columns=('index'))

# self.df_item.reset_index(drop=False, inplace=True)
# self.df_item.set_index(self.item_col, drop=True, inplace=True)
# self.df['ItemIdx'] = self.df_item.loc[self.df[self.item_col]]['index'].values
# self.df_item.drop(columns=('index'))
# u_inds = self.df_user[user_col].cat.codes.values
# i_inds = self.df_item[self.item_col].cat.codes.values
# self.df_user['NumRatings_Train'] = self._mat_train.sum(axis=1)[u_inds, 0]
# self.df_user['NumRatings_Test'] = self._mat_test.sum(axis=1)[u_inds, 0]
# self.df_item['NumRatings_Train'] = self._mat_train.sum(axis=0)[0, i_inds]
# self.df_item['NumRatings_Test'] = self._mat_test.sum(axis=0)[0, i_inds]

# def get_top_items_for_this_user(
#     self,
#     R_est,
#     user: int,
#     exclude_training: bool = True
# ):
#     """Returns top products for this user"""
#     user_pred =
#     items_for_this_user = R_est[user, :]
  # # trim the interaction data based on min user item conditions
        # ubool = (self.df_user[num_ratings_col] >= min_num_rating_per_user)
        # selected_users = self.df_user.loc[ubool, user_col]
        # ibool = (self.df_item[num_ratings_col] >= min_num_rating_per_item)
        # selected_items = self.df_item.loc[ibool, self.item_col]
        # user_bool = (self.df[user_col].isin(selected_users))
        # item_bool = (self.df[self.item_col].isin(selected_items))
        # self.df = self.df[item_bool & user_bool].reset_index(drop=True)
        # user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
        # self.df_user = self.df_user[user_bool]
        # item_bool = self.df_item[self.item_col].isin(self.df[self.item_col].unique())
        # self.df_item = self.df_item[item_bool]

        # # slice the user and item dataframes
        # iterator = zip([self.df_item, self.df_user], [self.item_col, user_col],
        #                [self.item_col, user_col])
        # for idf, id_colname, idx_colname in iterator:
        #     # idf = idf[idf[id_colname].isin(
        #     #     self.df[id_colname].unique())].copy()
        #     idf.sort_values(by=num_ratings_col, ascending=False, inplace=True)
        #     idf.reset_index(drop=True, inplace=True)
        #     idf.reset_index(drop=False, inplace=True)
        #     idf.set_index(id_colname, drop=True, inplace=True)
        #     self.df[idx_colname] = idf.loc[self.df[id_colname]]['index'].values
        #     idf.rename(columns={'index': idx_colname}, inplace=True)
        #     idf.reset_index(drop=False, inplace=True)
        #     idf.set_index(idx_colname, drop=True, inplace=True)

        #     self.user_col = 'user_idx'
        # self.item_col = 'item_idx'
        # num_ratings_col = 'num_interactions'
        # self.df = pd.DataFrame({
        #     self.user_col: users_index,
        #     self.item_col: items_index
        # }, dtype=int)

        # if timestamps is not None:
        #     self.df['Timestamp'] = timestamps

        # # create user and item dframes
        # agg_flag = {num_ratings_col: (self.item_col, 'count')}
        # self.df_user = self.df.groupby(self.user_col).agg(**agg_flag)
        # self.df_user.sort_values(by=num_ratings_col, inplace=True)
        # agg_flag = {num_ratings_col: (self.user_col, 'count')}
        # self.df_item = self.df.groupby(self.item_col).agg(**agg_flag)
        # self.df_item.sort_values(by=num_ratings_col, inplace=True)
