"""Base class for implementing matrix factorization"""
from dataclasses import dataclass
import sys
from functools import partial
import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit
from tqdm import tqdm
from pathos.pools import ProcessPool, ParallelPool
import multiprocessing as mp
from .utils import create_shared_memory_nparray, release_shared
# pylint: disable=invalid-name


@dataclass
class BPR:
    """
    Bayesian Personalized Ranking (BPR)
    """

    num_features: int  # number of features in user/item matrices
    num_iters: int  # number of iterations
    batch_size: int  # number of users selected per iteration, < num_users
    initial_std: float  # initial strength of gaussian dist to fill user/item
    reg_lambda: float  # regularization constant
    learning_rate: float  # learning rate fior sgd
    verbose: bool = False  # prints additional info when true

    # user item matrices
    umat_shm = None  # shared memory  object for saving user mat
    imat_shm = None  # shared memory object for saving item mat
    num_users = None
    num_items = None
    # iter_users = None  # user indices to iterate over

    def initiate(
        self,
        uimat,
        seed: int = 1234
    ):
        """Initiate fitting"""

        # # generate randomly sleetced user indices for trainin

        # Initialize user and item matrix
        self.num_users = uimat.shape[0]
        self.num_items = uimat.shape[1]

        # check if batch size is less than num_users
        if self.num_users < self.batch_size:
            self.batch_size = self.num_users
            print('WARNING: Batch size was assigned greater than # of users')
            print(f'Setting batch size to {self.num_users}\n')
        else:
            self.batch_size = self.batch_size

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

        # create shared memory objects for user and item matrix
        self.umat_shm = create_shared_memory_nparray(
            data=umat,
            name='umat',
        )
        self.imat_shm = create_shared_memory_nparray(
            data=imat,
            name='imat',
        )

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


def get_pos_neg_pair_for_this_user(
        iumat: spmatrix,
        iuser: int
):
    """Generate tuple of (user, pos_item, neg_item)"""
    assert isinstance(iumat, spmatrix), "Need a scipy sparse matrix!"
    num_users, num_items = iumat.shape
    assert iuser < num_users, "user index greater than # of users!"
    pos_items = iumat.indices[iumat.indptr[iuser]:iumat.indptr[iuser + 1]]
    while len(pos_items) == 0:  # if cant find pos interations, it swtiches users
        iuser = np.random.choice(num_users)
        pos_items = iumat.indices[iumat.indptr[iuser]:iumat.indptr[iuser + 1]]
    pos_item = np.random.choice(pos_items)
    neg_item = np.random.choice(num_items)
    while neg_item in pos_items:
        neg_item = np.random.choice(num_items)
    return (iuser, pos_item, neg_item)


def bpr_update_step(this_user, bpr_obj, iumat):
    """Update user and item matrix"""

    # extract columns for specific u, i, j
    user_u = bpr_obj.umat[this_user]
    _, item_ith, item_jth = get_pos_neg_pair_for_this_user(
        iumat, this_user)
    item_i = bpr_obj.imat[item_ith]
    item_j = bpr_obj.imat[item_jth]

    # actual computation
    r_uij = np.dot(user_u, (item_i - item_j))
    sigmoid = expit(r_uij)
    sigmoid_tiled = np.tile(sigmoid, (bpr_obj.num_features,))
    grad_u = np.multiply(sigmoid_tiled, (item_j - item_i))
    grad_u += bpr_obj.reg_lambda * user_u
    grad_i = np.multiply(sigmoid_tiled, -user_u) + \
        bpr_obj.reg_lambda * item_i
    grad_j = np.multiply(sigmoid_tiled, user_u) + \
        bpr_obj.reg_lambda * item_j
    bpr_obj.umat[this_user] -= bpr_obj.learning_rate * grad_u
    bpr_obj.imat[item_ith] -= bpr_obj.learning_rate * grad_i
    bpr_obj.imat[item_jth] -= bpr_obj.learning_rate * grad_j
    return 0


def bpr_fit(bpr_obj, iumat, ncores):
    """fit function"""
    user_list = bpr_obj.sample_users()
    user_list = np.array(user_list).ravel()
    pbar = tqdm(
        iterable=user_list,
        total=len(user_list),
        position=0,
        leave=True,
        file=sys.stdout,
        desc='Fit'
    )
    fn = partial(bpr_update_step, bpr_obj=bpr_obj, iumat=iumat)
    if ncores == 1:
        results = [fn(ix) for ix in user_list]
    else:
        # with pa.pools.ParallelPool(ncpus=ncores) as p:
        with mp.Pool(ncores) as p:
            results = p.map(fn, pbar)
    return results


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
