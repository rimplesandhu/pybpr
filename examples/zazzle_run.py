import sys
import os
import numpy as np
import pandas as pd
from pybpr import UserItemInteractions, BPR, bpr_fit

if __name__ == "__main__":

    # load the processed data
    DATA_DIR = '/projects/zazzle/rsandhu/pybpr/examples/output/zazzle_data'
    vdf = pd.read_parquet(os.path.join(DATA_DIR, 'view_data.parquet'))
    cdf = pd.read_parquet(os.path.join(DATA_DIR, 'click_data.parquet'))
    odf = pd.read_parquet(os.path.join(DATA_DIR, 'order_data.parquet'))

    # create useritem objects
    num_users = vdf['user_idx'].max()+1
    num_items = vdf['product_idx'].max()+1
    view_data = UserItemInteractions(
        name='ZAZZLE VIEW DATA',
        users_index=vdf['user_idx'],
        items_index=vdf['product_idx']
    )
    view_data.generate_train_test(user_test_ratio=0.0)

    click_data = UserItemInteractions(
        name='ZAZZLE CLICK DATA',
        users_index=cdf['user_idx'],
        items_index=cdf['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    click_data.generate_train_test(user_test_ratio=0.0)

    order_data = UserItemInteractions(
        name='ZAZZLE ORDER DATA',
        users_index=odf['user_idx'],
        items_index=odf['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    order_data.generate_train_test(user_test_ratio=0.0)

    # define functions for running multiple simulations

    def get_metric(this_data, this_bpr):
        """run it for this instance"""
        bpr_fit(
            bpr_obj=this_bpr,
            uimat=this_data.mat,
            ncores=104
        )
        imetric = this_bpr.get_metric_v1(
            uimat=this_data.mat,
            perc_active_users=0.75,
            perc_active_items=0.75,
            num_recs=60
        )
        return imetric

    def run_simulation(this_data, this_bpr, num_runs, name='run_test'):
        """runs simulation and saves output"""
        try:
            this_bpr.initiate(uimat=this_data.mat)
            metric_log = []
            for _ in range(num_runs):
                metric_log.append(get_metric(this_data, this_bpr))
                np.savetxt(os.path.join(DATA_DIR, f'{name}.txt'), metric_log)
        except Exception as _:
            print(f'Error in {name}')

    # define bpr pbject
    bpr_run = BPR(
        num_features=100,
        reg_lambda=0.0,
        num_iters=50,
        learning_rate=0.2,
        batch_size=10000,
        initial_std=0.0001,
    )

    # click data
    run_simulation(view_data, bpr_run, 100, 'run_view_lrate2')
    run_simulation(click_data, bpr_run, 100, 'run_click_lrate2')
    run_simulation(order_data, bpr_run, 100, 'run_order_lrate2')

    # dont forget to release shared moemory, NREL HPC HATES THIS!!
    bpr_run.release_shm()
