"""
Base class for implementing BPR

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""
import os
from dataclasses import dataclass
from typing import Callable
import sys
from functools import partial
import multiprocessing as mp
import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from scipy.special import expit
from tqdm import tqdm
# from pathos.pools import ProcessPool, ParallelPool
from .utils import create_shared_memory_nparray, release_shared
from .utils import get_indices_sorted_by_activity

# pylint: disable=invalid-name


@dataclass
class BPR:
    """
    Bayesian Personalized Ranking (BPR)
    """

    mname: str = 'bpr_model'  # name of the model
    num_features: int = 10  # number of features in user/item matrices
    num_iters: int = 100  # number of iterations
    batch_size: int = 32  # number of users selected per iteration, < num_users
    initial_std: float = 0.0001  # initial strength of gaussian dist to fill user/item
    reg_lambda: float = 0.0  # regularization constant
    learning_rate: float = 0.001  # learning rate fior sgd
    verbose: bool = False  # prints additional info when true

    # user item matrices
    umat_shm = None  # shared memory  object for saving user mat
    imat_shm = None  # shared memory object for saving item mat
    num_users = None
    num_items = None

    metric_tracker = None
    # iter_users = None  # user indices to iterate over

    def initiate(
        self,
        num_users: int,
        num_items: int,
        seed: int = 1234
    ):
        """Initiate fitting"""
        self.num_users = num_users
        self.num_items = num_items
        self._check_dims()

        # initiate user and item matrices
        np.random.seed(seed)
        umat = np.random.normal(
            loc=0.,
            scale=self.initial_std,
            size=(self.num_users, self.num_features)
        )
        imat = np.random.normal(
            loc=0.,
            scale=self.initial_std,
            size=(self.num_items, self.num_features)
        )
        self._create_shm(umat=umat, imat=imat)
        self.metric_tracker = []

    def _check_dims(self):
        """check validatity of num_users, num_items and batch_size"""
        # check if batch size is less than num_users
        if self.num_users < self.batch_size:
            self.batch_size = self.num_users
            print('WARNING: Batch size assigned > # of users')
            print(f'Setting batch size to {self.num_users}')
        else:
            self.batch_size = self.batch_size

    def _create_shm(self, umat: np.ndarray, imat: np.ndarray) -> None:
        """Create shared memory objects"""
        self.umat_shm = create_shared_memory_nparray(
            data=umat,
            name='umat',
        )
        self.imat_shm = create_shared_memory_nparray(
            data=imat,
            name='imat',
        )

    def save_model(self, dir_name: str) -> None:
        """Save the model"""
        print(f'Saving the model in {dir_name}', flush=True)
        np.savez(
            file=os.path.join(dir_name, self.mname),
            umat=self.umat,
            imat=self.imat
        )

    def load_model(self, dir_name: str) -> None:
        """Save the model"""
        print(f'Loading the model from {dir_name}', flush=True)
        npzfile = np.load(os.path.join(dir_name, f'{self.mname}.npz'))
        print(npzfile, npzfile['umat'].shape)
        umat = npzfile['umat']
        imat = npzfile['imat']
        self.num_users = umat.shape[0]
        self.num_items = imat.shape[0]
        self.num_features = umat.shape[1]
        self._create_shm(umat=umat, imat=imat)

    def get_recomendations_for_this_user(
        self,
        user_idx: int,
        num_recs: int = 1,
        # exclude_liked: bool = True
    ):
        """Returns top products for this user"""
        rec_mat = self.umat[user_idx].dot(self.imat.T)
        print(rec_mat.shape)
        top_rec_inds = np.argpartition(
            a=-1*rec_mat,  # the recomendation matrix, -1 to counter ascending sort
            kth=num_recs-1,  # where to partition
            # axis=1  # sort it along the column, i.e. for each user
        )
        # liked = set(
        #     self.mat_train[user_idx].indices) if exclude_liked else set()
        # top_n = islice(
        #     [ix for ix in top_inds if ix not in liked], int(num_items))
        # # top_val = islice([user_pred[ix]
        #                  for ix in top_inds if ix not in liked], num_items)
        return list(top_rec_inds[:num_recs])

    def sample_users(self):
        """creates a list of randonly selected users for all iterations"""
        smp_fn = partial(
            np.random.choice,
            a=self.num_users,
            size=self.batch_size,
            replace=False
        )
        user_list = [smp_fn() for _ in range(self.num_iters)]
        user_list = np.array(user_list).ravel()
        return user_list

    def get_metric_v1(
        self,
        uimat: spmatrix,  # user-item matrix
        perc_active_users: float,  # consider only top -- percentage of users
        perc_active_items: float,  # consider only top -- percentage of users
        num_recs: int,  # number of recomendations
        max_users_per_batch: int = 100,
        percentiles: list[int] = [0.25, 0.5, 0.75],
        seed: int = 12345

    ):
        """"Compute the metric"""
        # get the index of top --user_count-- users
        # print('Computing BPR metric v1..', end="", flush=True)
        ucount = np.unique(uimat.tocoo().row).size
        ucount = min(int(perc_active_users*self.num_users), ucount)
        selected_users = get_indices_sorted_by_activity(
            uimat=uimat,
            axis=1,
            count=ucount
        )
        # print(ucount, selected_users.shape)

        # icount = np.unique(uimat.tocoo().col).size
        # icount = min(int(perc_active_users*self.num_items), icount)
        # selected_items = get_indices_sorted_by_activity(
        #     uimat=uimat,
        #     axis=0,
        #     count=icount
        # )
        # print(icount, selected_items.shape)
        # imat_sliced = self.imat.take(selected_items, axis=0)

        rng = np.random.default_rng(seed)
        rng.shuffle(selected_users)  # shufle the user list
        num_splits = int(len(selected_users)/max_users_per_batch)+1
        user_looper = tqdm(
            iterable=np.array_split(selected_users, num_splits),
            total=num_splits,
            position=0,
            leave=True,
            file=sys.stdout,
            desc='BPR-Score'
        )
        out_perc = np.zeros(shape=(len(percentiles),))
        # print(out_perc)
        for _, sel_users in enumerate(user_looper):
            umat_sliced = self.umat.take(sel_users, axis=0)
            # print('umat sliced', umat_sliced.shape)
            rec_mat = umat_sliced.dot(self.imat.T)
            # print('rec_mat', rec_mat.shape)
            # get indices of top rec_count recomendations
            # argpartition does not care about the order but gets top n right
            # argpartition faster than argsort if we dont care about order in top n
            # num_recs = min(num_recs, len(selected_items))
            top_rec_inds = np.argpartition(
                a=-1*rec_mat,  # the recomendation matrix, -1 to counter ascending sort
                kth=num_recs-1,  # where to partition
                axis=1  # sort it along the column, i.e. for each user
            )
            # print('top_rec_inds, before', top_rec_inds.shape)
            top_rec_inds = top_rec_inds[:, :num_recs]
            # print('top rec inds, after', top_rec_inds.shape)
            # top_rec_inds = selected_items.take(top_rec_inds)
            # print('top rec inds, after2', top_rec_inds.shape)

            # create a dummy user-item matrix with the recs obtained above
            dummy_rec_mat = csr_matrix(
                (np.repeat(True, len(sel_users)*num_recs),
                 (np.repeat(sel_users, num_recs), np.ravel(top_rec_inds, order='C'))),
                dtype=bool,  # uimat.dtype,
                shape=(self.num_users, self.num_items)
            )
            # print('dummy rec', dummy_rec_mat.shape)

            newmat = (uimat > 0.01).astype(bool)
            # for each selected user, compute the recs present in either
            # the rec matrix or in ui matrix, not both
            nonoverlap_count = np.asarray((dummy_rec_mat-newmat).sum(axis=1))
            nonoverlap_count = nonoverlap_count.reshape(-1).take(sel_users)

            # For each selected user, compute the # of interactions present
            interaction_count = np.asarray(newmat.sum(axis=1))
            interaction_count = interaction_count.reshape(-1).take(sel_users)
            # print(interaction_count)

            # this gets us the perc of recs found in
            overlap_ratio = np.divide(
                interaction_count + num_recs - nonoverlap_count,
                2*np.minimum(interaction_count, num_recs),

            )
            # print(overlap_ratio)
            out_perc += np.quantile(overlap_ratio, percentiles)
        out_val = out_perc/num_splits
        self.metric_tracker.append(out_val)
        return out_val

    def get_metric_v2(
        self,
        uimat: spmatrix,  # user-item matrix
        perc_active_users: float,  # consider only top -- percentage of users
        perc_active_items: float,  # consider only top -- percentage of users
        num_recs: int,  # number of recomendations
        max_users_per_batch: int = 100,
        percentiles: list[int] = [0.25, 0.5, 0.75],
        seed: int = 12345

    ):
        """"Compute the metric"""
        # get the index of top --user_count-- users
        # print('Computing BPR metric v1..', end="", flush=True)
        ucount = np.unique(uimat.tocoo().row).size
        ucount = min(int(perc_active_users*self.num_users), ucount)
        selected_users = get_indices_sorted_by_activity(
            uimat=uimat,
            axis=1,
            count=ucount
        )

        rng = np.random.default_rng(seed)
        rng.shuffle(selected_users)  # shufle the user list
        num_splits = int(len(selected_users)/max_users_per_batch)+1
        user_looper = tqdm(
            iterable=np.array_split(selected_users, num_splits),
            total=num_splits,
            position=0,
            leave=True,
            file=sys.stdout,
            desc='BPR-Score'
        )
        out_perc = np.zeros(shape=(len(percentiles),))
        for _, sel_users in enumerate(user_looper):
            umat_sliced = self.umat.take(sel_users, axis=0)
            rec_mat = umat_sliced.dot(self.imat.T)

            # get indices of top rec_count recomendations
            # argpartition does not care about the order but gets top n right
            # argpartition faster than argsort if we dont care about order in top n
            # num_recs = min(num_recs, len(selected_items))
            top_rec_inds = np.argpartition(
                a=-1*rec_mat,  # the recomendation matrix, -1 to counter ascending sort
                kth=num_recs-1,  # where to partition
                axis=1  # sort it along the column, i.e. for each user
            )
            top_rec_inds = top_rec_inds[:, :num_recs]
            # top_rec_inds = selected_items.take(top_rec_inds)

            # create a dummy user-item matrix with the recs obtained above
            dummy_rec_mat = csr_matrix(
                (np.repeat(True, len(sel_users)*num_recs),
                 (np.repeat(sel_users, num_recs), np.ravel(top_rec_inds))),
                dtype=bool,
                shape=(self.num_users, self.num_items)
            )

            # for each selected user, compute the recs present in either
            # the rec matrix or in ui matrix, not both
            nonoverlap_count = np.asarray((dummy_rec_mat-uimat).sum(axis=1))
            nonoverlap_count = nonoverlap_count.reshape(-1).take(sel_users)

            # For each selected user, compute the # of interactions present
            interaction_count = np.asarray(uimat.sum(axis=1))
            interaction_count = interaction_count.reshape(-1).take(sel_users)

            # this gets us the perc of recs found in
            overlap_ratio = np.divide(
                interaction_count + num_recs - nonoverlap_count,
                2*np.minimum(interaction_count, num_recs),

            )
            out_perc += np.quantile(overlap_ratio, percentiles)
        out_val = out_perc/num_splits
        self.metric_tracker.append(out_val)
        return out_val

    def release_shm(self):
        """Release shared memory"""
        print('Releasing shared memory..', flush=True, end="")
        try:
            release_shared('umat')
            release_shared('imat')
            print('done')
        except FileNotFoundError as _:
            print('nothing to release')

    @property
    def umat(self):
        """Get user matrix"""
        umat = np.frombuffer(self.umat_shm.buf)
        return umat.reshape(self.num_users, self.num_features)

    @property
    def imat(self):
        """Get item matrix"""
        imat = np.frombuffer(self.imat_shm.buf)
        return imat.reshape(self.num_items, self.num_features)


def bpr_update(
        this_user: int,
        bpr_obj: BPR,
        negative_sampler: Callable[[int], tuple[int, int]]
):
    """Update user and item matrix"""

    # get pos-neg pair and extract the vectors for those pair
    item_ith, item_jth = negative_sampler(iuser=this_user)
    item_i = bpr_obj.imat[item_ith]
    item_j = bpr_obj.imat[item_jth]
    user_u = bpr_obj.umat[this_user]

    # actual computation
    # blockify this step so moeory isseus are resolved
    # create running average of mean and variance of metric
    # stop when the metric stops changing
    r_uij = np.dot(user_u, (item_i - item_j))

    # r_uij_new = np.sum(user_u * (item_i - item_j))
    sigmoid = expit(-r_uij)
    # sigmoid_new = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
    sigmoid_tiled = np.tile(sigmoid, (bpr_obj.num_features,))
    grad_u = np.multiply(sigmoid_tiled, (item_j - item_i))
    # grad_u *= scale(vision_embed)
    # grad_u *= scale(text_embed)
    # grad_u *= time_scaling
    grad_u += bpr_obj.reg_lambda * user_u
    grad_i = np.multiply(sigmoid_tiled, -user_u) + \
        bpr_obj.reg_lambda * item_i
    grad_j = np.multiply(sigmoid_tiled, user_u) + \
        bpr_obj.reg_lambda * item_j
    bpr_obj.umat[this_user] -= bpr_obj.learning_rate * grad_u
    bpr_obj.imat[item_ith] -= bpr_obj.learning_rate * grad_i
    bpr_obj.imat[item_jth] -= bpr_obj.learning_rate * grad_j
    # learning rate combination of weights from pos and neg int


def bpr_fit(bpr_obj, neg_sampler, ncores):
    """fit function"""
    user_list = bpr_obj.sample_users()
    user_list = np.array(user_list).ravel()
    pbar = tqdm(
        iterable=user_list,
        total=len(user_list),
        position=0,
        leave=True,
        file=sys.stdout,
        desc='BPR-Train'
    )
    fn = partial(
        bpr_update,
        bpr_obj=bpr_obj,
        negative_sampler=neg_sampler
    )
    if ncores == 1:
        _ = [fn(ix) for ix in pbar]
    else:
        # with ParallelPool(ncpus=ncores) as p:
        with mp.Pool(ncores) as p:
            p.map(fn, pbar)


# junk
    # def gather_pair_matrix(self):  # dont need to do this
    #     """Generate tuple of (user, pos_item, neg_item)"""
    #     loop = trange(self.num_iters, desc=self.__class__.__name__)
    #     list_of_users = []
    #     list_of_pos_items = []
    #     list_of_neg_items = []
    #     for _ in loop:
    #         for _ in range(self.batch_iters):
    #             sampled_users = np.random.choice(
    #                 a=self.num_users,
    #                 size=self.batch_size,
    #                 replace=False
    #             )
    #             list_of_users.extend(sampled_users.tolist())
    #             for _, user in enumerate(sampled_users):
    #                 pos_items = self.indices[self.indptr[user]                                             :self.indptr[user + 1]]
    #                 pos_item = np.random.choice(pos_items)
    #                 neg_item = np.random.choice(self.num_items)
    #                 while neg_item in pos_items:
    #                     neg_item = np.random.choice(self.num_items)
    #                 list_of_pos_items.append(pos_item)
    #                 list_of_neg_items.append(neg_item)
    #     zipper = zip(list_of_users, list_of_pos_items, list_of_neg_items)
    #     self.pairs = [(ij, ix, iy, iz)
    #                   for ij, (ix, iy, iz) in enumerate(zipper)]
    #     self.ndcg_shm = create_shared_memory_nparray(
    #         imat=np.zeros((len(self.pairs),)),
    #         iname='ndcg_vec',
    #         dtype=np.float64
    #     )

    # def gather_pairs(self):  # dont need to do this
    #     """Generate tuple of (user, pos_item, neg_item)"""
    #     loop = trange(self.num_iters, desc=self.__class__.__name__)
    #     list_of_users = []
    #     list_of_pos_items = []
    #     list_of_neg_items = []
    #     for _ in loop:
    #         for _ in range(self.batch_iters):
    #             sampled_users = np.random.choice(
    #                 a=self.num_users,
    #                 size=self.batch_size,
    #                 replace=False
    #             )
    #             list_of_users.extend(sampled_users.tolist())
    #             for _, user in enumerate(sampled_users):
    #                 pos_items = self.indices[self.indptr[user]                                             :self.indptr[user + 1]]
    #                 pos_item = np.random.choice(pos_items)
    #                 neg_item = np.random.choice(self.num_items)
    #                 while neg_item in pos_items:
    #                     neg_item = np.random.choice(self.num_items)
    #                 list_of_pos_items.append(pos_item)
    #                 list_of_neg_items.append(neg_item)
    #     zipper = zip(list_of_users, list_of_pos_items, list_of_neg_items)
    #     self.pairs = [(ij, ix, iy, iz)
    #                   for ij, (ix, iy, iz) in enumerate(zipper)]
    #     self.ndcg_shm = create_shared_memory_nparray(
    #         imat=np.zeros((len(self.pairs),)),
    #         iname='ndcg_vec',
    #         dtype=np.float64
    #     )

    # # def update_this(self, ituple):
    # #     """Update user and item matrix"""
    # #     user_uth, item_ith, item_jth = ituple
    # #     user_u = self.umat[user_uth]
    # #     item_i = self.imat[item_ith]
    # #     item_j = self.imat[item_jth]
    # #     r_uij = np.dot(user_u, (item_i - item_j))
    # #     # r_uij = np.sum(user_u * (item_i - item_j), axis=1)
    # #     # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
    # #     sigmoid = expit(r_uij)
    # #     sigmoid_tiled = np.tile(sigmoid, (self.num_features,))
    # #     grad_u = np.multiply(sigmoid_tiled, (item_j - item_i))
    # #     grad_u += self.reg_lambda * user_u
    # #     grad_i = np.multiply(sigmoid_tiled, -user_u) + self.reg_lambda * item_i
    # #     grad_j = np.multiply(sigmoid_tiled, user_u) + self.reg_lambda * item_j
    # #     self.umat[user_uth] -= self.learning_rate * grad_u
    # #     self.imat[item_ith] -= self.learning_rate * grad_i
    # #     self.imat[item_jth] -= self.learning_rate * grad_j

    # def fit_this(self, ncores, ndcg_fun=None):
    #     """fit function"""
    #     func = partial(
    #         update_shm,
    #         user_shm=self.user_shm,
    #         item_shm=self.item_shm,
    #         n_user=self.num_users,
    #         n_item=self.num_items,
    #         n_feat=self.num_features,
    #         rlambda=self.reg_lambda,
    #         lrate=self.learning_rate,
    #         active_users=self.most_active_users,
    #         ndcg_shm=self.ndcg_shm,
    #         t_mat=self.tmat
    #     )
    #     pbar = tqdm(
    #         iterable=self.pairs,
    #         total=len(self.pairs),
    #         position=0,
    #         leave=True,
    #         file=sys.stdout
    #     )
    #     with pa.pools.ParallelPool(ncpus=ncores) as p:
    #         p.map(func, pbar)
    #         # p.imap_unordered(func, pbar)

    # def fit(
    #         self,
    #         ndcg_func=None
    # ):
    #     """
    #     Fit function
    #     """
    #     self.ndcg_metric = []
    #     loop = trange(self.num_iters, desc=self.__class__.__name__)
    #     for i in loop:
    #         for j in range(self.batch_iters):
    #             sampled_pos_items = np.zeros(self.batch_size, dtype=np.int)
    #             sampled_neg_items = np.zeros(self.batch_size, dtype=np.int)
    #             sampled_users = np.random.choice(
    #                 a=self.num_users,
    #                 size=self.batch_size,
    #                 replace=False  # a user cannot be repeated in each batch
    #             )
    #             for idx, user in enumerate(sampled_users):
    #                 pos_items = self.indices[self.indptr[user]
    #                     :self.indptr[user + 1]]
    #                 pos_item = np.random.choice(pos_items)
    #                 # if train_mat_neg is not None:
    #                 #     neg_items = indicesn[indptrn[user]:indptrn[user + 1]]
    #                 #     neg_item = np.random.choice(neg_items)
    #                 # else:
    #                 neg_item = np.random.choice(self.num_items)
    #                 while neg_item in pos_items:
    #                     neg_item = np.random.choice(self.num_items)
    #                 sampled_pos_items[idx] = pos_item
    #                 sampled_neg_items[idx] = neg_item
    #                 if self.verbose:
    #                     print(f'{i}-{j}: u{user}, p{pos_item}, n{neg_item}')
    #             # print(sampled_users, sampled_pos_items, sampled_neg_items)
    #             user_u = self.umat[sampled_users]
    #             item_i = self.imat[sampled_pos_items]
    #             item_j = self.imat[sampled_neg_items]
    #             r_uij = np.sum(user_u * (item_i - item_j), axis=1)
    #             # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
    #             sigmoid = expit(r_uij)
    #             sigmoid_tiled = np.tile(sigmoid, (self.num_features, 1)).T
    #             grad_u = sigmoid_tiled * \
    #                 (item_j - item_i) + self.reg_lambda * user_u
    #             grad_i = sigmoid_tiled * -user_u + self.reg_lambda * item_i
    #             grad_j = sigmoid_tiled * user_u + self.reg_lambda * item_j
    #             self.umat[sampled_users] -= self.learning_rate * grad_u
    #             self.imat[sampled_pos_items] -= self.learning_rate * grad_i
    #             self.imat[sampled_neg_items] -= self.learning_rate * grad_j

    #         if (ndcg_func is not None) & (i % self.ndcg_skip == 0) & (i != 0):
    #             # ndcg_test = ndcg_func(
    #             #     umat=self.umat,
    #             #     imat=self.imat,
    #             #     test=True
    #             # )
    #             # print('i am computing ndcg')
    #             ndcg_test = 0
    #             ndcg_train = ndcg_func(
    #                 umat=self.umat,
    #                 imat=self.imat,
    #                 test=False
    #             )
    #             self.ndcg_metric.append(
    #                 {'test': ndcg_test, 'train': ndcg_train}
    #             )

    # # def _sample(self, batch_size, indices, indptr):
    # #     """sample batches of random triplets u, i, j"""
    # #     sampled_pos_items = np.zeros(batch_size, dtype=np.int)
    # #     sampled_neg_items = np.zeros(batch_size, dtype=np.int)
    # #     sampled_users = np.random.choice(
    # #         self.num_users, size=batch_size, replace=False)

    # #     for idx, user in enumerate(sampled_users):
    # #         pos_items = indices[indptr[user]:indptr[user + 1]]
    # #         pos_item = np.random.choice(pos_items)
    # #         neg_item = np.random.choice(self.num_items)
    # #         while neg_item in pos_items:
    # #             neg_item = np.random.choice(self.num_items)
    # #         sampled_pos_items[idx] = pos_item
    # #         sampled_neg_items[idx] = neg_item

    # #     return sampled_users, sampled_pos_items, sampled_neg_items

    # def update(self, u, i, j):
    #     """
    #     update according to the bootstrapped user u,
    #     positive item i and negative item j
    #     """
    #     # print('i m here1')
    #     user_u = self.umat[u]
    #     item_i = self.imat[i]
    #     item_j = self.imat[j]

    #     # decompose the estimator, compute the difference between
    #     # the score of the positive items and negative items; a
    #     # naive implementation might look like the following:
    #     # r_ui = np.diag(user_u.dot(item_i.T))
    #     # r_uj = np.diag(user_u.dot(item_j.T))
    #     # r_uij = r_ui - r_uj

    #     # however, we can do better, so
    #     # for batch dot product, instead of doing the dot product
    #     # then only extract the diagonal element (which is the value
    #     # of that current batch), we perform a hadamard product,
    #     # i.e. matrix element-wise product then do a sum along the column will
    #     # be more efficient since it's less operations
    #     # http://people.revoledu.com/kardi/tutorial/LinearAlgebra/HadamardProduct.html
    #     # r_ui = np.sum(user_u * item_i, axis = 1)
    #     #
    #     # then we can achieve another speedup by doing the difference
    #     # on the positive and negative item up front instead of computing
    #     # r_ui and r_uj separately, these two idea will speed up the operations
    #     # from 1:14 down to 0.36
    #     r_uij = np.sum(user_u * (item_i - item_j), axis=1)
    #     # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
    #     sigmoid = expit(r_uij)
    #     # repeat the 1 dimension sigmoid n_factors times so
    #     # the dimension will match when doing the update
    #     sigmoid_tiled = np.tile(sigmoid, (self.num_features, 1)).T

    #     # update using gradient descent
    #     grad_u = sigmoid_tiled * (item_j - item_i) + self.reg_lambda * user_u
    #     grad_i = sigmoid_tiled * -user_u + self.reg_lambda * item_i
    #     grad_j = sigmoid_tiled * user_u + self.reg_lambda * item_j
    #     self.umat[u] -= self.learning_rate * grad_u
    #     self.imat[i] -= self.learning_rate * grad_i
    #     self.imat[j] -= self.learning_rate * grad_j
    # def _get_pair_for_this_user(self, iuser: int):  # dont need to do this
    #     """Generate tuple of (user, pos_item, neg_item)"""
    #     indices = np.frombuffer(self.rmat_indices_shm.buf).astype(int)
    #     indptr = np.frombuffer(self.rmat_indptr_shm.buf).astype(int)
    #     pos_items = indices[indptr[iuser]:indptr[iuser + 1]]
    #     pos_item = np.random.choice(pos_items)
    #     neg_item = np.random.choice(self.num_items)
    #     while neg_item in pos_items:
    #         neg_item = np.random.choice(self.num_items)
    #     return (pos_item, neg_item)
