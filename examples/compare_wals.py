"""WALS script"""
import os
import time
import itertools
import tqdm
import numpy as np
import pandas as pd
from functools import partial
from pybpr import MF_WALS, UserItemInteractions, load_movielens_data
import scipy.sparse as ss
import pathos.multiprocessing as mp
#import multiprocessing as mp


NCORES = 16


def get_ndcg_wals(irow, dfshort):
    idx, idict = irow
    print(idx, end="-", flush=True)
    cf = UserItemInteractions(
        name='MovieLens-100k',
        users=dfshort['user_id'],
        items=dfshort['item_id'],
        min_num_rating_per_user=20,
        min_num_rating_per_item=10
    )
    # cf.print_memory_usage()
    cf.generate_train_test(user_test_ratio=idict['test_ratio'])
    wals = MF_WALS(
        num_features=idict['num_features'],
        reg_lambda=idict['reg_lambda'],
        weighting_strategy=idict['wgt_strategy'],
        num_iters=idict['num_iters'],
        initial_std=0.1,
        seed=None
    )
    wals.fit(cf.R_train)
    wals_ndcg_func = partial(
        cf.get_ndcg_metric,
        user_mat=wals.user_mat,
        item_mat=wals.item_mat,
        num_items=idict['ndcg_num_items']
    )
    out_dict = {}
    out_dict['ndcg_test'] = wals_ndcg_func(test=True)
    out_dict['ndcg_train'] = wals_ndcg_func(test=False)
    return out_dict


if __name__ == "__main__":

    df = load_movielens_data('ml-100k')
    dfshort = df[df['rating'] > 0]

    # create combination of hyperparameters to vary
    list_of_num_features = np.arange(1, 100, 4)
    list_of_wgt_strategies = ['same', 'uniform',
                              'user-oriented', 'item-oriented']
    list_of_ndcg_num_items = [5, 10, 20]
    list_of_num_iters = [5, 10, 20]
    list_of_reg_lambda = [0.]
    list_of_test_ratio = [0.1, 0.2]

    iter_list = list(itertools.product(
        list_of_num_features,
        list_of_wgt_strategies,
        list_of_ndcg_num_items,
        list_of_num_iters,
        list_of_reg_lambda,
        list_of_test_ratio
    ))
    columns = ['num_features', 'wgt_strategy',
               'ndcg_num_items', 'num_iters', 'reg_lambda',
               'test_ratio']
    df = pd.DataFrame(list(iter_list), columns=columns)
    pfunc = partial(get_ndcg_wals, dfshort=dfshort)
    print(f'Got {df.shape[0]} hyperparameter combinations', flush=True)

    # serial
    start_time = time.time()
    results = []
    for irow in df.iterrows():
        results.append(pfunc(irow))

    # parallel
    # start_time = time.time()
    # with mp.ProcessPool(NCORES) as p:
    #     results = tqdm.tqdm(p.imap(pfunc, df.iterrows()), total=df.shape[0])

    # concat
    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    df.to_csv(os.path.join(os.path.curdir, 'output', 'ml100k_wals_results.csv'))
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)

    # df['ndcg_test'] = np.nan
    # df['ndcg_train'] = np.nan
    # df['wall_time'] = np.nan
