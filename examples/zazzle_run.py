import sys
import os
from functools import partial
import numpy as np
import pandas as pd
from pybpr import UserItemInteractions, BPR, bpr_fit
from pybpr import uniform_negative_sampler, explicit_negative_sampler

if __name__ == "__main__":

    # load the processed data
    DATA_DIR = '/projects/zazzle/rsandhu/pybpr/examples/output/zazzle_data'
    df_v = pd.read_parquet(os.path.join(DATA_DIR, 'view_data.parquet'))
    df_c = pd.read_parquet(os.path.join(DATA_DIR, 'click_data.parquet'))
    df_o = pd.read_parquet(os.path.join(DATA_DIR, 'order_data.parquet'))
    df_v_not_c = pd.read_parquet(os.path.join(
        DATA_DIR, 'viewed_not_clicked_data.parquet'))
    df_c_not_o = pd.read_parquet(os.path.join(
        DATA_DIR, 'clicked_not_ordered_data.parquet'))

    test_ratio = 0.25

    # viewed
    num_users = df_v['user_idx'].max()+1
    num_items = df_v['product_idx'].max()+1
    data_viewed = UserItemInteractions(
        users_index=df_v['user_idx'],
        items_index=df_v['product_idx']
    )
    data_viewed.generate_train_test(user_test_ratio=test_ratio)

    # viewed not clicked
    data_viewed_not_clicked = UserItemInteractions(
        users_index=df_v_not_c['user_idx'],
        items_index=df_v_not_c['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    data_viewed_not_clicked.generate_train_test(user_test_ratio=test_ratio)

    # clicked
    data_clicked = UserItemInteractions(
        users_index=df_c['user_idx'],
        items_index=df_c['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    data_clicked.generate_train_test(user_test_ratio=test_ratio)

    # clicked not ordered
    data_clicked_not_ordered = UserItemInteractions(
        users_index=df_c_not_o['user_idx'],
        items_index=df_c_not_o['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    data_clicked_not_ordered.generate_train_test(user_test_ratio=test_ratio)

    # ordered
    data_ordered = UserItemInteractions(
        users_index=df_o['user_idx'],
        items_index=df_o['product_idx'],
        num_users=num_users,
        num_items=num_items
    )
    data_ordered.generate_train_test(user_test_ratio=test_ratio)

    # define functions for running multiple simulatio
    def run_simulation(
        pos_data: UserItemInteractions,
        this_bpr: BPR,
        num_runs: int,
        neg_data: UserItemInteractions = None,
        name='run_test'
    ):
        """runs simulation and saves output"""
        this_bpr.initiate(
            num_users=pos_data.num_users,
            num_items=pos_data.num_items
        )
        if neg_data is not None:
            neg_sampler = partial(
                explicit_negative_sampler,
                pos_uimat=pos_data.mat_train,
                neg_uimat=neg_data.mat
            )
        else:
            neg_sampler = partial(
                uniform_negative_sampler,
                uimat=pos_data.mat_train
            )

        try:
            metric_log = []
            for _ in range(num_runs):
                bpr_fit(
                    bpr_obj=this_bpr,
                    neg_sampler=neg_sampler,
                    ncores=104
                )
                imetric = this_bpr.get_metric_v1(
                    uimat=pos_data.mat_test,
                    perc_active_users=0.95,
                    perc_active_items=0.75,
                    num_recs=60
                )
                metric_log.append(imetric)
                np.savetxt(os.path.join(DATA_DIR, f'{name}.txt'), metric_log)
        except Exception as _:
            print(f'Error in {name}')

    # define bpr pbject
    bpr_run = BPR(
        num_features=200,
        reg_lambda=0.0,
        num_iters=500,
        learning_rate=0.1,
        batch_size=15000,
        initial_std=0.0001,
    )
    num_runs = 50

    # simulations

    run_simulation(
        pos_data=data_clicked,
        neg_data=data_viewed_not_clicked,
        this_bpr=bpr_run,
        num_runs=num_runs,
        name='run3_pos_clicked_neg_viewed_not_clicked_test'
    )

    run_simulation(
        pos_data=data_clicked,
        neg_data=None,
        this_bpr=bpr_run,
        num_runs=num_runs,
        name='run3_pos_clicked_test'
    )

    run_simulation(
        pos_data=data_ordered,
        neg_data=data_clicked_not_ordered,
        this_bpr=bpr_run,
        num_runs=num_runs,
        name='run3_pos_ordered_neg_clicked_not_ordered_test'
    )

    run_simulation(
        pos_data=data_ordered,
        neg_data=None,
        this_bpr=bpr_run,
        num_runs=num_runs,
        name='run3_pos_ordered_test'
    )

    # run_simulation(
    #     pos_data=data_viewed,
    #     neg_data=None,
    #     this_bpr=bpr_run,
    #     num_runs=num_runs,
    #     name='run3_pos_viewed'
    # )

    # dont forget to release shared moemory, NREL HPC HATES THIS!!
    bpr_run.release_shm()
