"""WALS script"""
import os
import time
import itertools
import tqdm
import numpy as np
import pandas as pd
from functools import partial
from pybpr import MF_WALS, UserItemInteractions, load_movielens_data, BPR
import scipy.sparse as ss
import pathos.multiprocessing as mp
#import multiprocessing as mp


def get_ndcg_bpr(irow, dfshort):
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
    bpr = BPR(
        num_features=idict['num_features'],
        reg_lambda=idict['reg_lambda'],
        num_iters=idict['num_iters'],
        learning_rate=idict['learning_rate'],
        batch_size=idict['batch_size'],
        initial_std=idict['initial_std'],
        seed=None
    )
    out_dict = {}
    try:
        start_this = time.time()
        bpr.fit(cf.R_train)
        bpr_ndcg_func = partial(
            cf.get_ndcg_metric,
            user_mat=bpr.user_mat,
            item_mat=bpr.item_mat,
            num_items=int(idict['ndcg_num_items'])
        )
        out_dict = {}
        out_dict['ndcg_test'] = bpr_ndcg_func(test=True)
        out_dict['ndcg_train'] = bpr_ndcg_func(test=False)
        out_dict['wall_time'] = time.time() - start_this
    except:
        out_dict['ndcg_test'] = np.nan
        out_dict['ndcg_train'] = np.nan
        out_dict['wall_time'] = np.nan
    return out_dict


if __name__ == "__main__":

    df = load_movielens_data('ml-100k')
    dfshort = df[df['rating'] > 0]
    hyperparameter_choices = {
        'num_features': np.arange(2, 80, 2),
        'ndcg_num_items': [10],
        'num_iters': [200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.25, 0.5],
        'reg_lambda': [0., 2., 5.],
        'batch_size': [5, 10, 20],
        'initial_std': [0.01],
        'test_ratio': [0.2]
    }
    iter_list = list(itertools.product(*hyperparameter_choices.values()))
    rdf = pd.DataFrame(iter_list, columns=hyperparameter_choices.keys())
    pfunc = partial(get_ndcg_bpr, dfshort=dfshort)
    print(f'Got {rdf.shape[0]} hyperparameter combinations', flush=True)

    # serial
    # start_time = time.time()
    # results = []
    # for irow in rdf.iterrows():
    #     results.append(pfunc(irow))

    # parallel
    start_time = time.time()
    with mp.ProcessPool(8) as p:
        results = p.map(pfunc, rdf.iterrows())

    rdf = pd.concat([rdf, pd.DataFrame(results)], axis=1)
    rdf.to_csv(os.path.join(os.path.curdir, 'output', 'ml100k_bpr_results.csv'))
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)
