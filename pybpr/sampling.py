"""
Negative sampling 
Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

import numpy as np
from scipy.sparse import spmatrix
from datetime import datetime, timedelta
from scipy.special import expit


def uniform_negative_sampler(
        iuser: int,
        uimat: spmatrix
):
    """Generate tuple of (user, pos_item, neg_item)"""

    # check if valid sparse matrix
    assert isinstance(uimat, spmatrix), "Need a scipy sparse matrix!"
    num_users, num_items = uimat.shape
    assert iuser < num_users, "user index greater than # of users!"

    # if cant find pos interations, it swtiches user
    pos_items = uimat.indices[uimat.indptr[iuser]:uimat.indptr[iuser + 1]]
    while len(pos_items) == 0:
        iuser = np.random.choice(num_users)
        pos_items = uimat.indices[uimat.indptr[iuser]:uimat.indptr[iuser + 1]]
    pos_item = np.random.choice(pos_items)

    # uniform negative sampling
    neg_item = np.random.choice(num_items)
    while neg_item in pos_items:
        neg_item = np.random.choice(num_items)
    return (pos_item, neg_item)


def explicit_negative_sampler(
        iuser: int,
        pos_uimat: spmatrix,
        neg_uimat: spmatrix
):
    """Generate tuple of (user, pos_item, neg_item)"""

    # check if valid sparse matrix
    assert isinstance(pos_uimat, spmatrix), "Need a scipy sparse matrix!"
    assert isinstance(neg_uimat, spmatrix), "Need a scipy sparse matrix!"
    assert pos_uimat.shape == neg_uimat.shape, 'Need compatible sparse mats!'
    num_users, num_items = pos_uimat.shape
    assert iuser < num_users, "user index greater than # of users!"

    # sample pos interaction
    pos_items = pos_uimat.indices[
        pos_uimat.indptr[iuser]:pos_uimat.indptr[iuser + 1]
    ]
    while len(pos_items) == 0:
        iuser = np.random.choice(num_users)
        pos_items = pos_uimat.indices[
            pos_uimat.indptr[iuser]:pos_uimat.indptr[iuser + 1]
        ]
    pos_item = np.random.choice(pos_items)

    # sample negative interaction
    neg_items = neg_uimat.indices[
        neg_uimat.indptr[iuser]:neg_uimat.indptr[iuser + 1]
    ]
    if len(neg_items) == 0:
        neg_item = np.random.choice(num_items)
        while neg_item in pos_items:
            neg_item = np.random.choice(num_items)
    else:
        neg_item = np.random.choice(neg_items)

    return (pos_item, neg_item)


def time_explicit_negative_sampler(
        iuser: int,
        pos_uimat: spmatrix,
        neg_uimat: spmatrix
):
    """Generate tuple of (user, pos_item, neg_item)"""

    # check if valid sparse matrix
    assert isinstance(pos_uimat, spmatrix), "Need a scipy sparse matrix!"
    assert isinstance(neg_uimat, spmatrix), "Need a scipy sparse matrix!"
    assert pos_uimat.shape == neg_uimat.shape, 'Need compatible sparse mats!'
    num_users, num_items = pos_uimat.shape
    assert iuser < num_users, "user index greater than # of users!"

    # sample pos interaction
    pos_items = pos_uimat.indices[
        pos_uimat.indptr[iuser]:pos_uimat.indptr[iuser + 1]
    ]
    if len(pos_items) == 0:
        pos_valid_users = np.unique(pos_uimat.tocoo().row)
        iuser = np.random.choice(pos_valid_users)
        pos_items = pos_uimat.indices[
            pos_uimat.indptr[iuser]:pos_uimat.indptr[iuser + 1]
        ]
    pos_wgts = pos_uimat.tocsc()[iuser, pos_items].toarray()[0, :]
    # while len(pos_items) == 0:
    #     iuser = np.random.choice(num_users)
    #     pos_items = pos_uimat.indices[
    #         pos_uimat.indptr[iuser]:pos_uimat.indptr[iuser + 1]
    #     ]
    pos_item = np.random.choice(pos_items, p=pos_wgts/np.sum(pos_wgts))

    # sample negative interaction
    neg_items = neg_uimat.indices[
        neg_uimat.indptr[iuser]:neg_uimat.indptr[iuser + 1]
    ]
    if len(neg_items) == 0:
        neg_item = np.random.choice(num_items)
        while neg_item in pos_items:
            neg_item = np.random.choice(num_items)
    else:
        neg_wgts = neg_uimat.tocsc()[iuser, neg_items].toarray()[0, :]
        neg_item = np.random.choice(neg_items, p=neg_wgts/np.sum(neg_wgts))

    return (pos_item, neg_item)


def get_time_weighting_v0(
        x: int,
        datetime_median: datetime,
        datetime_cutoff: datetime,
        scaling_days: int
) -> float:
    scale = timedelta(days=scaling_days).total_seconds()
    time_cutoff = datetime_cutoff.timestamp()
    bool_cutoff = (x < time_cutoff).astype(float)
    time_median = datetime_median.timestamp()
    return np.multiply(expit((x-time_median)/scale), bool_cutoff)


def get_time_weighting(
        x: int,
        datetime_min: datetime,
        datetime_max: datetime,
        scaling_days: int = -1
) -> float:
    """Get time weigting"""
    time_min = datetime_min.timestamp()
    time_max = datetime_max.timestamp()
    bool_min = (x > time_min).astype(float)
    bool_max = (x < time_max).astype(float)
    out_val = np.multiply(bool_min, bool_max)
    scale = timedelta(days=scaling_days).total_seconds()
    time_middle = int((time_min+time_max)/2)
    wgts = expit((x-time_middle)/scale)
    out_val = np.multiply(wgts, out_val)
    return out_val

# viewed not clicked at the time of click is more important
# clicked not ordered at the time of order is more important
# further way to narrow your negative sampling space
