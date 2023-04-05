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


def get_ndcg_wals(irow, dfshort):
    _, idict = irow
    #print(idx, end="-", flush=True)
    cf = UserItemInteractions(
        name='MovieLens-100k',
        users=dfshort['user_id'],
        items=dfshort['item_id'],
        min_num_rating_per_user=10,
        min_num_rating_per_item=10
    )
    # cf.print_memory_usage()
    cf.generate_train_test(user_test_ratio=idict['test_ratio'])
    wals = MF_WALS(
        num_features=idict['num_features'],
        reg_lambda=idict['reg_lambda'],
        weighting_strategy=idict['wgt_strategy'],
        num_iters=idict['num_iters'],
        initial_std=idict['initial_std'],
        seed=None
    )
    out_dict = {}
    try:
        wals.fit(cf.R_train)
        wals_ndcg_func = partial(
            cf.get_ndcg_metric,
            user_mat=wals.user_mat,
            item_mat=wals.item_mat,
            num_items=idict['ndcg_num_items']
        )
        out_dict['ndcg_test'] = wals_ndcg_func(test=True)
        out_dict['ndcg_train'] = wals_ndcg_func(test=False)
    except:
        out_dict['ndcg_test'] = np.nan
        out_dict['ndcg_train'] = np.nan
    return out_dict


if __name__ == "__main__":

    df = load_movielens_data('ml-100k')
    dfshort = df[df['rating'] > 0]
    wgt_strategies = ['same', 'uniformly-negative',
                      'user-oriented', 'item-oriented']
    hyperparameter_choices = {
        'num_features': np.arange(2, 60, 3),
        'wgt_strategy': wgt_strategies,
        'ndcg_num_items': [10],
        'num_iters': [20, 30],
        'reg_lambda': [0., 2., 5.],
        'initial_std': [0.01],
        'test_ratio': [0.2],
    }
    iter_list = list(itertools.product(*hyperparameter_choices.values()))
    df = pd.DataFrame(iter_list, columns=hyperparameter_choices.keys())
    pfunc = partial(get_ndcg_wals, dfshort=dfshort)
    print(f'Got {df.shape[0]} hyperparameter combinations', flush=True)

    # serial
    start_time = time.time()
    results = []
    for irow in df.iterrows():
        results.append(pfunc(irow))

    # parallel
    # start_time = time.time()
    # with mp.ProcessPool(8) as p:
    #     results = tqdm.tqdm(p.imap(pfunc, df.iterrows()), total=df.shape[0])

    df = pd.concat([df, pd.DataFrame(results)], axis=1)
    df.to_csv(os.path.join(os.path.curdir, 'output', 'ml100k_wals_results.csv'))
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)
