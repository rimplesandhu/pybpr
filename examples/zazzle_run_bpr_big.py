#!/usr/bin/env python
# coding: utf-8

# ## Application of BPR on Zazzle Data

import os
import pandas as pd
import numpy as np
from functools import partial
from functools import reduce
from pybpr import *
from subprocess import call


# load data
DATA_DIR = '/kfs2/projects/zazzle/rsandhu/pybpr/examples/data/zazzle_big'
df_v = pd.read_parquet(os.path.join(DATA_DIR, 'view_data.parquet'))
df_c = pd.read_parquet(os.path.join(DATA_DIR, 'click_data.parquet'))
df_o = pd.read_parquet(os.path.join(DATA_DIR, 'order_data.parquet'))
df_v_not_c = pd.read_parquet(os.path.join(
    DATA_DIR, 'viewed_not_clicked_data.parquet'))
df_c_not_o = pd.read_parquet(os.path.join(
    DATA_DIR, 'clicked_not_ordered_data.parquet'))


# define useritem objects
user_colname = 'user_member_idx'
item_colname = 'product_idx'
num_users = df_v[user_colname].nunique()
num_items = df_v[item_colname].nunique()


test_ratio = 0.0
data_viewed_not_clicked = UserItemInteractions(
    users_index=df_v_not_c[user_colname],
    items_index=df_v_not_c[item_colname],
    num_users=num_users,
    num_items=num_items
)
data_viewed_not_clicked.generate_train_test(user_test_ratio=test_ratio)

# clicked
data_clicked = UserItemInteractions(
    users_index=df_c[user_colname],
    items_index=df_c[item_colname],
    num_users=num_users,
    num_items=num_items
)
data_clicked.generate_train_test(user_test_ratio=test_ratio)

# clicked not ordered
data_clicked_not_ordered = UserItemInteractions(
    users_index=df_c_not_o[user_colname],
    items_index=df_c_not_o[item_colname],
    num_users=num_users,
    num_items=num_items
)
data_clicked_not_ordered.generate_train_test(user_test_ratio=test_ratio)

# ordered
data_ordered = UserItemInteractions(
    users_index=df_o[user_colname],
    items_index=df_o[item_colname],
    num_users=num_users,
    num_items=num_items
)
data_ordered.generate_train_test(user_test_ratio=test_ratio)


# BPR
mybpr = BPR(
    num_features=500,
    reg_lambda=0.0,
    num_iters=500,
    learning_rate=0.1,
    batch_size=15000,
    initial_std=0.0001,
)
mybpr.initiate(num_users=num_users, num_items=num_items)


# training
pos_data = data_ordered
neg_data = data_clicked_not_ordered
metric_log_train = []
# neg_sampler = partial(
#     uniform_negative_sampler,
#     uimat=training_data
# )


neg_sampler = partial(
    explicit_negative_sampler,
    pos_uimat=pos_data.mat,
    neg_uimat=neg_data.mat
)

for _ in range(10):
    bpr_fit(
        bpr_obj=mybpr,
        neg_sampler=neg_sampler,
        ncores=104
    )
    mfunc = partial(
        mybpr.get_metric_v1,
        perc_active_users=0.1,
        perc_active_items=0.1,
        num_recs=60,
        max_users_per_batch=2000,
        percentiles=[0.25, 0.5, 0.75],
        seed=1234
    )
    iscore = mfunc(uimat=pos_data.mat)
    metric_log_train.append(iscore)
    print('Score:', [np.round(ix, 4) for ix in iscore])
metric_log_train = np.asarray(metric_log_train)


# Save output
OUT_DIR = '/projects/zazzle/rsandhu/pybpr/examples/output'
np.savetxt(os.path.join(OUT_DIR, 'score_log1.txt'), metric_log_train)
mybpr.save_model(dir_name=OUT_DIR)
mybpr.release_shm()
