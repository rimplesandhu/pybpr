"""Utilities"""
# pylint: disable=invalid-name

import os
from typing import Callable, List
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.sparse import spmatrix, csr_matrix
import scipy.sparse as sp
from numpy import ndarray


def get_most_active_users_from_uimat(uimat, count: int | None = None):
    """return the most -count- active users"""
    # get number of interactions for each user
    count_vector = np.asarray(uimat.sum(axis=1)).reshape(-1)
    # get the index of users with most interactions at 0
    sorted_user_list = count_vector.argsort()[::-1]
    if count is not None:
        sorted_user_list = sorted_user_list[:count]
    return sorted_user_list


def compute_metric(
        umat: ndarray,  # user matrix
        imat: ndarray,  # item matrix
        uimat: spmatrix,  # user item sparse matrix
        user_count: int,  # number of top users
        num_recs: int
):
    """"Compute the metric"""
    num_users, num_items = uimat.shape
    active_users = get_most_active_users_from_uimat(uimat, user_count)
    umat_sliced = umat.take(active_users, axis=0)
    rec_mat = umat_sliced.dot(imat.T)
    top_recs = np.argsort(rec_mat)[:, ::-1][:, :num_recs]
    rec_user_inds = np.repeat(active_users, num_recs)
    rec_item_inds = np.ravel(top_recs)
    rec_R = csr_matrix(
        (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
        dtype=np.int8,
        shape=(num_users, num_items)
    )
    rinds, _, _ = sp.find(rec_R + uimat)
    # rinds, _, _ = sp.find(rec_R - rec_mat)
    # n_rec = rec_R[self.most_active_users].count_nonzero()
    # n_int = rec_mat[self.most_active_users].count_nonzero()
    # ndcg_vec = np.frombuffer(ndcg_shm.buf)
    # ndcg_vec = (n_rec + n_int - rinds.size)/n_user
    return user_count*num_recs-(len(rinds)-uimat.nnz)


# def update_shm(
#     ituple, user_shm, item_shm,
#     n_user, n_item, n_feat, rlambda,
#     lrate, active_users, ndcg_shm, t_mat
# ):
#     """Update user and item matrix"""
#     user_mat = np.frombuffer(user_shm.buf).reshape(
#         n_user, n_feat)  # only need a specific row
#     item_mat = np.frombuffer(item_shm.buf).reshape(
#         n_item, n_feat)  # only need a specific row

#     ith, user_uth, item_ith, item_jth = ituple
#     user_u = user_mat[user_uth]
#     item_i = item_mat[item_ith]
#     item_j = item_mat[item_jth]
#     r_uij = np.dot(user_u, (item_i - item_j))
#     # compute this along with ndcg
#     # r_uij = np.sum(user_u * (item_i - item_j), axis=1)
#     # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
#     sigmoid = expit(r_uij)
#     sigmoid_tiled = np.tile(sigmoid, (n_feat,))
#     grad_u = np.multiply(sigmoid_tiled, (item_j - item_i))
#     grad_u += rlambda * user_u
#     grad_i = np.multiply(sigmoid_tiled, -user_u) + rlambda * item_i
#     grad_j = np.multiply(sigmoid_tiled, user_u) + rlambda * item_j
#     user_mat[user_uth] -= lrate * grad_u
#     item_mat[item_ith] -= lrate * grad_i
#     item_mat[item_jth] -= lrate * grad_j

#     if ith == 10000:
#         print(ith, flush=True)
#         rec_mat = user_mat.take(active_users, axis=0).dot(item_mat.T)
#         top_recs = np.argsort(rec_mat)[:, ::-1][:, :60]
#         rec_user_inds = np.repeat(active_users, 60)
#         rec_item_inds = np.ravel(top_recs)
#         rec_R = csr_matrix(
#             (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
#             dtype=np.int8,
#             shape=(n_user, n_item)
#         )
#         rinds, _, _ = sp.find(rec_R[active_users] - t_mat[active_users])
#         n_rec = rec_R[active_users].count_nonzero()
#         n_int = t_mat[active_users].count_nonzero()
#         ndcg_vec = np.frombuffer(ndcg_shm.buf)
#         ndcg_vec[ith] = (n_rec + n_int - rinds.size)/n_user

#     #     ndcg_vec
#     #     # batch dot product
#     #     print(mp.current_process().name, len(active_users))
#     # rec_mat = user_mat.take(active_users, axis=0).dot(
#     #     item_mat.T)  # 5000*num_items
#     # top_recs = np.argsort(rec_mat)[:, ::-1][:, :60] # 5000*60
#     # rec_user_inds = np.repeat(active_users, 60)
#     # rec_item_inds = np.ravel(top_recs)
#     # rec_R = csr_matrix(
#     #     (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
#     #     dtype=np.int8,
#     #     shape=(n_user, n_item)
#     # )
#     # rinds, _, _ = sp.find(rec_R[active_users] - t_mat[active_users])
#     # n_rec = rec_R[active_users].count_nonzero()
#     # n_int = t_mat[active_users].count_nonzero()
#     # out = (n_rec + n_int - rinds.size)/n_user


def create_shared_memory_nparray(
    data: ndarray,
    name: str,
    dtype=np.float64
):
    """Create shared memory object"""
    d_size = np.dtype(dtype).itemsize * np.prod(data.shape)
    shm = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    except FileExistsError:
        release_shared(name)
        shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    finally:
        dst = np.ndarray(shape=data.shape, dtype=dtype, buffer=shm.buf)
        dst[:] = data[:]
    return shm


def release_shared(name):
    """Release shared memory block"""
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()  # Free and release the shared memory block


def compute_mse(y_true, y_pred):
    """ignore zero terms prior to comparing the mse"""
    mask = np.nonzero(y_true)
    assert mask[0].shape[0] > 0, 'Truth matrix empty'
    mse = mean_squared_error(np.array(y_true[mask]).ravel(), y_pred[mask])
    return mse


def load_movielens_data(flag='ml-100k'):
    """Function to read movielens data"""
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    if flag == 'ml-100k':
        file_path = os.path.join(os.path.curdir, 'data', flag, 'u.data')
        df = pd.read_csv(file_path, sep='\t', names=names)
        return df
    elif flag == 'ml-1m':
        file_path = os.path.join(os.path.curdir, 'data', flag, 'ratings.dat')
        df = pd.read_csv(file_path, sep='::', names=names, engine='python')
        return df
    elif flag == 'ml-10M100K':
        file_path = os.path.join(os.path.curdir, 'data', flag, 'ratings.dat')
        df = pd.read_csv(file_path, sep='::', names=names, engine='python')
        return df
    else:
        raise ValueError('Choose among ml-100k, ml-1m, ml-10M100K')


def compute_ndcg(
    ranked_item_idx: List[int],
    K: int,
    wgt_fun: Callable = np.log2
):
    """Comutes NDCG metric"""
    assert K > 0, 'Should have atleast one recomendation, choose K >1'
    ndcg_score = 0.
    if np.array(ranked_item_idx).size > 0:
        assert np.max(
            ranked_item_idx) < K, 'entry in ranked_item_idx > K-1!'
        Rup = np.zeros(K, dtype=int)
        Rup[ranked_item_idx] = 1.
        wgt = np.array([1 / wgt_fun(ix + 1) for ix in np.arange(1, K + 1)])
        # wgt = np.array([1. for ix in np.arange(1, K + 1)])
        ndcg_score = np.sum(np.multiply(wgt, Rup)) / np.sum(wgt)
    return ndcg_score


def get_interaction_weights(
    train_mat,
    strategy: str = 'same',
    fac: float | None = None
):
    """Function for getting the weights"""
    row_inds, col_inds, _ = ss.find(train_mat)
    num_users, num_items = train_mat.shape
    match strategy.lower():
        case 'positive-only':
            weight_mat = np.zeros(train_mat.shape)
            weight_mat[row_inds, col_inds] = 1.
        case 'uniformly-negative':
            weight_mat = np.random.uniform(size=train_mat.shape)
            weight_mat[row_inds, col_inds] = 1.
        case 'user-oriented':
            fac = np.amax(train_mat.sum(axis=1)) if fac is None else fac
            weight_mat = np.clip(train_mat.sum(axis=1), 0, fac) / fac
            weight_mat = np.array(np.repeat(weight_mat, num_items, axis=1))
            weight_mat[row_inds, col_inds] = 1.
        case 'item-oriented':
            fac = np.amax(train_mat.sum(axis=0)) if fac is None else fac
            weight_mat = 1 - np.clip(train_mat.sum(axis=0), 0, fac) / fac
            weight_mat = np.array(np.repeat(weight_mat, num_users, axis=0))
            weight_mat[row_inds, col_inds] = 1.
        case _:
            weight_mat = np.ones(train_mat.shape)
    return weight_mat
