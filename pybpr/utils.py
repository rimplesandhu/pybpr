#!/usr/bin/env python3
"""Utility functions for data preprocessing."""
import json
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def get_nonempty_rows_csr(csr_mat: csr_matrix) -> np.ndarray:
    """Get row indices with nonzero entries - O(num_users)."""
    return np.where(np.diff(csr_mat.indptr) > 0)[0]


def get_user_interactions(
    users: List[int],
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    neg_ratio: float = 1.,
    exclude_pos_csr: Optional[csr_matrix] = None,
) -> list[tuple]:
    """Collect positive and negative interactions per user."""
    assert neg_ratio > 0., 'Need neg_ratio>0'
    n_users, n_items = pos_csr_mat.shape

    # Pre-compute indices for efficient CSR access
    pos_indptr = pos_csr_mat.indptr
    pos_indices = pos_csr_mat.indices
    neg_indptr = neg_csr_mat.indptr
    neg_indices = neg_csr_mat.indices

    # Full positives for neg exclusion (avoid train leakage)
    if exclude_pos_csr is not None:
        excl_indptr = exclude_pos_csr.indptr
        excl_indices = exclude_pos_csr.indices
    else:
        excl_indptr = pos_indptr
        excl_indices = pos_indices

    user_interactions = []

    for user_idx in users:
        # Get positive items for this user
        pos_start = pos_indptr[user_idx]
        pos_end = pos_indptr[user_idx + 1]
        pos_items = pos_indices[pos_start:pos_end]

        if len(pos_items) == 0:
            continue

        # Get or generate negative items
        neg_start = neg_indptr[user_idx]
        neg_end = neg_indptr[user_idx + 1]
        neg_items = neg_indices[neg_start:neg_end]

        # Handle case with no explicit negative items
        if len(neg_items) == 0:
            # All known positives to exclude
            excl_start = excl_indptr[user_idx]
            excl_end = excl_indptr[user_idx + 1]
            excl_items = set(
                excl_indices[excl_start:excl_end]
            )

            # Number of negatives to sample
            n_neg = int(neg_ratio * len(pos_items))
            n_neg = min(n_neg, n_items - len(excl_items))
            if n_neg <= 0:
                continue

            # Sample negatives excluding all positives
            neg_items = np.empty(n_neg, dtype=np.int32)
            sampled = 0
            while sampled < n_neg:
                batch_size = min(
                    n_neg - sampled, n_neg
                )
                candidates = np.random.randint(
                    0, n_items, size=batch_size
                )
                for item in candidates:
                    if item not in excl_items:
                        neg_items[sampled] = item
                        sampled += 1
                        if sampled >= n_neg:
                            break

        user_interactions.append(
            (user_idx, pos_items, neg_items)
        )

    return user_interactions


def compute_auc_scores(
    user_interactions: list[tuple],
    predict_fn: callable
) -> list[float]:
    """Calculate ROC AUC scores using user interactions."""
    all_scores = []

    for user_idx, pos_items, neg_items in user_interactions:
        # Prepare item arrays for batch prediction
        all_items = np.concatenate([pos_items, neg_items])

        # Get scores for all items at once
        all_predictions = predict_fn(user_idx, all_items)

        # Prepare labels (1 for positive, 0 for negative)
        y_true = np.zeros(len(all_items), dtype=np.int8)
        y_true[:len(pos_items)] = 1

        # Compute AUC for this user
        user_auc = roc_auc_score(y_true, all_predictions)
        all_scores.append(user_auc)

    return all_scores


def sample_pos_neg_pairs(
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    user_indices: Optional[List[int]] = None,
    exclude_pos_csr: Optional[csr_matrix] = None,
    random_state: Optional[Union[int, RandomState]] = None
) -> Tuple[List[int], List[int], List[int]]:
    """Sample positive-negative item pairs for users."""
    # Input validation
    if pos_csr_mat.nnz == 0:
        raise ValueError("Positive matrix is empty")

    if pos_csr_mat.shape != neg_csr_mat.shape:
        raise ValueError(
            f"Shape mismatch: {pos_csr_mat.shape} vs "
            f"{neg_csr_mat.shape}")

    # Initialize random state for reproducibility
    rng = (
        RandomState(random_state)
        if not isinstance(random_state, RandomState)
        else random_state
    )

    # Process user indices with validation
    n_users = pos_csr_mat.shape[0]
    if user_indices is None:
        user_indices = list(range(n_users))
    else:
        # Validate user indices are within bounds
        invalid_indices = [
            idx for idx in user_indices
            if idx < 0 or idx >= n_users
        ]
        if invalid_indices:
            raise IndexError(
                f"User indices out of bounds: "
                f"{invalid_indices}"
            )

    # Convert to LIL format for efficient row access
    pos_lil_mat = pos_csr_mat.tolil()
    neg_lil_mat = neg_csr_mat.tolil()

    # Use full positives for neg exclusion if provided
    exclude_lil = (
        exclude_pos_csr.tolil()
        if exclude_pos_csr is not None
        else pos_lil_mat
    )

    # Initialize result arrays
    valid_user_indices = []
    pos_item_indices = []
    neg_item_indices = []
    num_items = pos_csr_mat.shape[1]

    # Process users
    for user_idx in user_indices:
        pos_items = pos_lil_mat.rows[user_idx]

        if len(pos_items) == 0:
            continue

        # Sample positive item
        pos_idx = pos_items[rng.randint(0, len(pos_items))]

        # Try to sample from explicit negatives first
        neg_items = neg_lil_mat.rows[user_idx]
        if len(neg_items) > 0:
            neg_idx = neg_items[
                rng.randint(0, len(neg_items))
            ]
        else:
            # Exclude all known positives (not just split)
            exclude_items = exclude_lil.rows[user_idx]
            neg_idx = rng.choice(num_items)
            while neg_idx in exclude_items:
                neg_idx = rng.choice(num_items)

        valid_user_indices.append(user_idx)
        pos_item_indices.append(pos_idx)
        neg_item_indices.append(neg_idx)

    return (
        valid_user_indices,
        pos_item_indices,
        neg_item_indices,
    )


def load_json_file(filepath: str) -> dict:
    """Load and parse a JSON file from the given filepath."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filepath}': {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading '{filepath}': {e}")
        raise


def split_sparse_coo_matrix(
    matrix: sparse.coo_matrix,
    train_ratio: float,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    show_progress: bool = False,
) -> Tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Split sparse COO matrix into train and test sets."""
    if not isinstance(matrix, sparse.coo_matrix):
        raise TypeError("Input matrix must be a scipy.sparse.coo_matrix")

    if not 0 <= train_ratio <= 1:
        raise ValueError(
            "train_ratio must be between 0 and 1 inclusive")

    # Handle edge cases
    if train_ratio == 0:
        return sparse.coo_matrix(matrix.shape), matrix
    if train_ratio == 1:
        return matrix, sparse.coo_matrix(matrix.shape)

    # Set random state
    rng = (
        random_state if isinstance(
            random_state, np.random.RandomState)
        else np.random.RandomState(random_state)
    )

    nnz = matrix.nnz
    rows, cols, data = matrix.row, matrix.col, matrix.data

    # Initial random split
    idx = np.arange(nnz)
    rng.shuffle(idx)
    n_train = int(nnz * train_ratio)

    is_train = np.zeros(nnz, dtype=bool)
    is_train[idx[:n_train]] = True

    # Find users present in data but missing from training set
    unique_users = np.unique(rows)
    users_in_train = np.unique(rows[is_train])
    missing_users = np.setdiff1d(unique_users, users_in_train)

    if len(missing_users) > 0:
        # Move one random interaction to train for each missing user
        if show_progress:
            missing_user_iter = tqdm(
                missing_users,
                desc="Ensuring user representation"
            )
        else:
            missing_user_iter = missing_users

        for user in missing_user_iter:
            # Find test indices for this user
            user_test_mask = (rows == user) & ~is_train
            user_test_indices = np.where(user_test_mask)[0]

            if len(user_test_indices) > 0:
                # Move one random interaction to train
                chosen_idx = rng.choice(user_test_indices)
                is_train[chosen_idx] = True

    # Create train and test matrices
    train_idx = np.where(is_train)[0]
    test_idx = np.where(~is_train)[0]

    train_matrix = sparse.coo_matrix(
        (data[train_idx], (rows[train_idx], cols[train_idx])),
        shape=matrix.shape
    )
    test_matrix = sparse.coo_matrix(
        (data[test_idx], (rows[test_idx], cols[test_idx])),
        shape=matrix.shape
    )

    return train_matrix, test_matrix


def split_sparse_coo_matrix_old(
    matrix: sparse.coo_matrix,
    train_ratio: float,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    show_progress: bool = False,
) -> Tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Split sparse COO matrix ensuring all rows/cols in training."""
    if not isinstance(matrix, sparse.coo_matrix):
        raise TypeError("Input matrix must be a scipy.sparse.coo_matrix")

    if not 0 <= train_ratio <= 1:
        raise ValueError(
            "train_ratio must be between 0 and 1 inclusive")

    # Handle edge cases for train_ratio
    if train_ratio == 0:
        return sparse.coo_matrix(matrix.shape), matrix

    if train_ratio == 1:
        return matrix, sparse.coo_matrix(matrix.shape)

    # Set random state
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    # Get matrix dimensions and non-zero entries
    n_rows, n_cols = matrix.shape
    nnz = matrix.nnz
    rows = matrix.row
    cols = matrix.col
    data = matrix.data

    # Create a shuffled index array
    idx = np.arange(nnz)
    rng.shuffle(idx)

    # Calculate number of entries for training set
    n_train = int(nnz * train_ratio)

    # Initial split
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    # Check if all rows and columns are represented in training
    train_rows = set(rows[train_idx])
    train_cols = set(cols[train_idx])

    # Find missing rows and columns in training set
    missing_rows = set(range(n_rows)) - train_rows
    missing_cols = set(range(n_cols)) - train_cols

    # Move entries to train for each missing row
    if show_progress and missing_rows:
        missing_row_iter = tqdm(
            missing_rows, desc="Train/Test Split (Rows)")
    else:
        missing_row_iter = missing_rows

    for row in missing_row_iter:
        # Find entries for this row that are in the test set
        row_entries = []
        for i in test_idx:
            if rows[i] == row:
                row_entries.append(i)

        if row_entries:
            # Choose a random entry for this row
            entry_idx = rng.choice(row_entries)

            # Move from test to train
            test_idx = test_idx[test_idx != entry_idx]
            train_idx = np.append(train_idx, entry_idx)

    # Move entries to train for each missing column
    if show_progress and missing_cols:
        missing_col_iter = tqdm(
            missing_cols, desc="Train/Test Split (Cols)")
    else:
        missing_col_iter = missing_cols

    for col in missing_col_iter:
        # Find entries for this column that are in the test set
        col_entries = []
        for i in test_idx:
            if cols[i] == col:
                col_entries.append(i)

        if col_entries:
            # Choose a random entry for this column
            entry_idx = rng.choice(col_entries)

            # Move from test to train
            test_idx = test_idx[test_idx != entry_idx]
            train_idx = np.append(train_idx, entry_idx)

    # Create train and test matrices
    train_matrix = sparse.coo_matrix(
        (data[train_idx], (rows[train_idx], cols[train_idx])),
        shape=matrix.shape
    )

    test_matrix = sparse.coo_matrix(
        (data[test_idx], (rows[test_idx], cols[test_idx])),
        shape=matrix.shape
    )

    return train_matrix, test_matrix


def get_sparse_matrix_stats(matrix: sparse.coo_matrix) -> dict:
    """Get basic statistics about a sparse matrix."""
    stats = {
        "shape": matrix.shape,
        "nnz": matrix.nnz,
        "density": matrix.nnz / max(
            (matrix.shape[0] * matrix.shape[1]), 1),
        "empty_rows": matrix.shape[0]-len(np.unique(matrix.row)),
        "empty_cols": matrix.shape[1]-len(np.unique(matrix.col)),
    }
    return stats


def print_sparse_matrix_stats(matrix: sparse.coo_matrix) -> str:
    """Print compact stats about a sparse matrix in a single line."""
    stats = get_sparse_matrix_stats(matrix)
    print_str = (
        f"({stats['shape'][0]:5}×{stats['shape'][1]:5}) "
        f"nnz={stats['nnz']:9,} ({stats['density']:5.3%}),"
        f"empty={stats['empty_rows']:5}/{stats['empty_cols']:5}"
    )
    return print_str


def series_to_categorical_int(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to categorical integers."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    return series.astype('category').cat.codes


def get_category_indices(
    series1: pd.Series,
    series2: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Align categories between two pandas Series."""
    # Get union of all unique values from both series
    all_categories = pd.Index(
        sorted(
            set(series1.dropna().unique()) |
            set(series2.dropna().unique())
        )
    )

    # Convert both series to categorical with the same categories
    cat_series1 = pd.Categorical(series1, categories=all_categories)
    cat_series2 = pd.Categorical(series2, categories=all_categories)

    # Get codes (indices) for each categorical series
    codes_series1 = pd.Series(cat_series1.codes, index=series1.index)
    codes_series2 = pd.Series(cat_series2.codes, index=series2.index)

    return codes_series1, codes_series2


def scipy_csr_to_torch_csr(scipy_csr: csr_matrix) -> torch.Tensor:
    """Convert a SciPy sparse CSR matrix to PyTorch sparse CSR."""
    # Get the CSR components from the SciPy matrix
    data = scipy_csr.data
    indices = scipy_csr.indices
    indptr = scipy_csr.indptr
    shape = scipy_csr.shape

    # Convert numpy arrays to torch tensors
    data_torch = torch.from_numpy(data)
    indices_torch = torch.from_numpy(indices).to(torch.int64)
    indptr_torch = torch.from_numpy(indptr).to(torch.int64)

    # Create the PyTorch sparse CSR tensor
    return torch.sparse_csr_tensor(
        crow_indices=indptr_torch,
        col_indices=indices_torch,
        values=data_torch,
        size=shape
    )


def slice_scipy_to_torch_sparse(
    scipy_matrix: sp.spmatrix,
    row_indices: Union[List[int], np.ndarray]
) -> torch.Tensor:
    """Slice a SciPy sparse matrix by row indices to PyTorch."""
    # Convert to CSR for efficient row slicing if needed
    if not sp.isspmatrix_csr(scipy_matrix):
        scipy_matrix = scipy_matrix.tocsr()

    # Slice the matrix by row indices
    sliced_matrix = scipy_matrix[row_indices, :]

    # Convert to COO format (coordinate format)
    coo_matrix = sliced_matrix.tocoo()

    # Extract coordinates and values
    indices = torch.LongTensor(
        np.vstack((coo_matrix.row, coo_matrix.col)))
    values = torch.FloatTensor(coo_matrix.data)
    shape = torch.Size(coo_matrix.shape)

    # Create PyTorch sparse tensor
    return torch.sparse_coo_tensor(indices, values, shape)
