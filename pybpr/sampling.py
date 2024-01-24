"""
Negative sampling 
Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

import numpy as np
from scipy.sparse import spmatrix


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
