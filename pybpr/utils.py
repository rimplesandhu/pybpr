
#!/usr/bin/env python3
"""
Utility functions for data preprocessing and categorical data handling.
"""

import os
import math
from typing import List, Union, Optional, Tuple
import scipy.sparse as sp
from scipy import sparse
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import json
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed, cpu_count


def get_user_interactions(
    users: List[int],
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    neg_ratio: float = 1.
) -> list[tuple]:
    """Collect positive and negative interactions for each user.

    Args:
        pos_csr_mat: Sparse matrix of positive interactions
        neg_csr_mat: Sparse matrix of negative interactions
        neg_ratio: Ratio to determine number of negative samples

    Returns:
        List of tuples (user_idx, pos_items, neg_items) for each user
    """
    assert neg_ratio > 0., 'Need neg_ratio>0'
    n_users, n_items = pos_csr_mat.shape

    # Pre-compute indices for efficient CSR access
    pos_indptr = pos_csr_mat.indptr
    pos_indices = pos_csr_mat.indices
    neg_indptr = neg_csr_mat.indptr
    neg_indices = neg_csr_mat.indices

    user_interactions = []

    for user_idx in users:
        pos_start, pos_end = pos_indptr[user_idx], pos_indptr[user_idx + 1]
        pos_items = pos_indices[pos_start:pos_end]

        if len(pos_items) == 0:
            continue

        # Get or generate negative items
        neg_start, neg_end = neg_indptr[user_idx], neg_indptr[user_idx + 1]
        neg_items = neg_indices[neg_start:neg_end]

        # Handle case with no negative items
        if len(neg_items) == 0:
            # get number of neg items
            n_neg_items = int(neg_ratio * len(pos_items))
            n_neg_items = min(n_neg_items, n_items - len(pos_items))
            if n_neg_items <= 0:
                continue

            # get negative items
            neg_items = np.empty(n_neg_items, dtype=np.int32)
            pos_items_set = set(pos_items)
            sampled = 0
            while sampled < n_neg_items:
                batch_size = min(n_neg_items - sampled, n_neg_items)
                candidates = np.random.randint(0, n_items, size=batch_size)
                for item in candidates:
                    if item not in pos_items_set:
                        neg_items[sampled] = item
                        sampled += 1
                        if sampled >= n_neg_items:
                            break

        user_interactions.append((user_idx, pos_items, neg_items))

    return user_interactions


def compute_auc_scores(
    user_interactions: list[tuple],
    predict_fn: callable
) -> list[float]:
    """Calculate ROC AUC scores using user interactions and a prediction function.

    Args:
        user_interactions: List of (user_idx, pos_items, neg_items) tuples
        predict_fn: Function that takes (user_idx, items) and returns prediction scores
                    for all items at once

    Returns:
        List of AUC scores for each user
    """
    all_scores = []

    for user_idx, pos_items, neg_items in user_interactions:
        # Prepare item arrays for batch prediction
        all_items = np.concatenate([pos_items, neg_items])

        # Get scores for all items at once
        all_predictions = predict_fn(user_idx, all_items)

        # Prepare labels
        y_true = np.zeros(len(all_items), dtype=np.int8)
        y_true[:len(pos_items)] = 1

        # Compute AUC
        # print(user_idx, all_predictions.shape, y_true.shape)
        user_auc = roc_auc_score(y_true, all_predictions).item()
        all_scores.append(user_auc)
        # all_scores.append((y_true, all_predictions, user_auc))

    return all_scores


# def compute_roc_auc_summary(auc_scores: list[float]) -> tuple[float, float]:
#     """Compute mean and standard deviation of AUC scores.

#     Args:
#         auc_scores: List of AUC scores

#     Returns:
#         Tuple of (mean AUC score, standard deviation)
#     """
#     if not auc_scores:
#         return 0.0, 0.0

#     auc_scores = np.array(auc_scores)
#     return float(np.mean(auc_scores)), float(np.std(auc_scores))


# def compute_roc_auc_score_all_items(
#     pred_scores: np.ndarray,
#     pos_csr_mat: csr_matrix,
#     neg_csr_mat: csr_matrix,
#     neg_ratio: float = 1.,
#     n_jobs: int = -1
# ) -> tuple[float, float]:
#     """Calculate ROC AUC score for recommendation system predictions.

#     Args:
#         pred_scores: Matrix of prediction scores (n_users, n_items)
#         pos_csr_mat: Sparse matrix of positive interactions
#         neg_csr_mat: Sparse matrix of negative interactions
#         neg_ratio: Ratio to determine number of negative samples
#         n_jobs: Number of parallel jobs (-1 uses all cores)

#     Returns:
#         Tuple of (mean AUC score, standard deviation) across users
#     """
#     # Dynamically calculate batch size based on number of jobs
#     assert neg_ratio > 0., 'Need neg_ratio>0'
#     n_users, n_items = pred_scores.shape
#     n_jobs = n_jobs if n_jobs > 0 else cpu_count()
#     optimal_batch_size = max(
#         min(math.ceil(n_users / (n_jobs * 4)), 1000),
#         50
#     )

#     # Pre-compute indices for efficient CSR access
#     pos_indptr = pos_csr_mat.indptr
#     pos_indices = pos_csr_mat.indices
#     neg_indptr = neg_csr_mat.indptr
#     neg_indices = neg_csr_mat.indices

#     def process_user_batch(user_batch):
#         batch_scores = []
#         for user_idx in user_batch:
#             pos_start, pos_end = pos_indptr[user_idx], pos_indptr[user_idx + 1]
#             pos_items = pos_indices[pos_start:pos_end]

#             if len(pos_items) == 0:
#                 continue

#             # Get or generate negative items
#             neg_start, neg_end = neg_indptr[user_idx], neg_indptr[user_idx + 1]
#             neg_items = neg_indices[neg_start:neg_end]

#             # Handle case with no negative items
#             if len(neg_items) == 0:
#                 # get number of neg items
#                 n_neg_items = int(neg_ratio * len(pos_items))
#                 n_neg_items = min(n_neg_items, n_items - len(pos_items))
#                 if n_neg_items <= 0:
#                     continue

#                 # get negative items
#                 neg_items = np.empty(n_neg_items, dtype=np.int32)
#                 pos_items_set = set(pos_items)
#                 sampled = 0
#                 while sampled < n_neg_items:
#                     batch_size = min(n_neg_items - sampled, n_neg_items)
#                     candidates = np.random.randint(0, n_items, size=batch_size)
#                     for item in candidates:
#                         if item not in pos_items_set:
#                             neg_items[sampled] = item
#                             sampled += 1
#                             if sampled >= n_neg_items:
#                                 break

#             # Collect prediction scores in vectorized operations
#             n_total = len(pos_items) + len(neg_items)
#             y_score = np.empty(n_total, dtype=pred_scores.dtype)
#             y_score[:len(pos_items)] = pred_scores[user_idx, pos_items]
#             y_score[len(pos_items):] = pred_scores[user_idx, neg_items]

#             # Prepare labels and scores as a single block
#             y_true = np.zeros(n_total, dtype=np.int8)
#             y_true[:len(pos_items)] = 1

#             # compute
#             user_auc = roc_auc_score(y_true, y_score)
#             batch_scores.append(user_auc)

#         return batch_scores

#     # Split users into batches for parallel processing
#     user_batches = []
#     for start_idx in range(0, n_users, optimal_batch_size):
#         end_idx = min(start_idx + optimal_batch_size, n_users)
#         user_batches.append(range(start_idx, end_idx))

#     # Process user batches in parallel
#     all_scores = []
#     with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
#         batch_results = parallel(
#             delayed(process_user_batch)(batch) for batch in user_batches
#         )

#         # Flatten the batch results
#         for batch_scores in batch_results:
#             all_scores.extend(batch_scores)

#     return all_scores


# def compute_roc_auc_score(pred_scores, pos_csr_mat, neg_csr_mat, neg_ratio):
#     """auc"""
#     pos_lil_mat = pos_csr_mat.tolil()
#     neg_lil_mat = neg_csr_mat.tolil()
#     n_users, n_items = pred_scores.shape

#     for user_idx in range(n_users):
#         # pos items
#         pos_items = pos_lil_mat.rows[user_idx]
#         if len(pos_items) == 0:
#             continue

#         # neg items
#         max_neg_items = n_items-len(pos_items)
#         neg_items = neg_lil_mat.rows[user_idx]
#         if len(neg_items) == 0:
#             n_neg_items = min(int(neg_ratio*len(pos_items)), max_neg_items)
#             neg_items = np.random.choice(n_items, size=n_neg_items)
#             inds = np.where(np.isin(neg_items, pos_items))[0]
#             while len(inds) == 0:
#                 neg_items[inds] = np.random.choice(n_items, size=len(inds))
#                 inds = np.where(np.isin(neg_items, pos_items))[0]


# def compute_user_roc_auc(
#     pos_scores: Union[List[float], np.ndarray],
#     neg_scores: Union[List[float], np.ndarray]
# ) -> np.ndarray:
#     """Compute ROC AUC when each user has positive and negative scores.

#     Parameters:
#         pos_scores: Array of positive scores per user
#         neg_scores: Array of negative scores per user

#     Returns:
#         individual_aucs
#     """
#     # Convert inputs to numpy arrays
#     pos_scores = np.asarray(pos_scores)
#     neg_scores = np.asarray(neg_scores)

#     # Validate inputs
#     if len(pos_scores) != len(neg_scores):
#         raise ValueError("Score arrays must have the same length")
#     if len(pos_scores) == 0:
#         return np.array([])

#     # Create mask for non-zero scores
#     valid_mask = (pos_scores != 0) & (neg_scores != 0)
#     individual_aucs = np.full(len(pos_scores), np.nan)

#     # Initialize AUC scores array
#     individual_aucs = np.zeros(len(pos_scores))

#     # Compute comparisons for vectorized assignment
#     pos_greater = pos_scores > neg_scores
#     pos_equal = pos_scores == neg_scores

#     # Assign AUC values based on comparisons (1.0 or 0.5)
#     individual_aucs[valid_mask & pos_greater] = 1.0
#     individual_aucs[valid_mask & pos_equal] = 0.5

#     return individual_aucs


def sample_pos_neg_pairs(
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    user_indices: Optional[List[int]] = None,
    explicit_only: bool = False,
    random_state: Optional[Union[int, RandomState]] = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Sample positive-negative item pairs for users.

    Args:
        pos_csr_mat: Positive user-item interactions sparse matrix.
        neg_csr_mat: Negative user-item interactions sparse matrix.
        user_indices: User indices to process (all users if None).
        explicit_only: Only sample from explicit negative matrix if True.
        random_state: Random seed or RandomState object.

    Returns:
        Tuple of (valid_user_indices, pos_item_indices, neg_item_indices).

    Raises:
        ValueError: If matrices are empty or shapes don't match.
        IndexError: If user_indices contains out-of-bounds indices.
    """
    # Input validation
    if pos_csr_mat.nnz == 0:
        raise ValueError("Positive matrix is empty")

    if pos_csr_mat.shape != neg_csr_mat.shape:
        raise ValueError(f"Shape mismatch: {pos_csr_mat.shape} vs "
                         f"{neg_csr_mat.shape}")

    # Initialize random state for reproducibility
    rng = (RandomState(random_state) if not isinstance(random_state, RandomState)
           else random_state)

    # Process user indices with validation
    n_users = pos_csr_mat.shape[0]
    if user_indices is None:
        user_indices = list(range(n_users))
    else:
        # Validate user indices are within bounds
        invalid_indices = [
            idx for idx in user_indices if idx < 0 or idx >= n_users]
        if invalid_indices:
            raise IndexError(f"User indices out of bounds: {invalid_indices}")

    # Convert to LIL format for efficient row access
    pos_lil_mat = pos_csr_mat.tolil()
    neg_lil_mat = neg_csr_mat.tolil()

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
            neg_idx = neg_items[rng.randint(0, len(neg_items))]
        else:
            # Sample from implicit negatives (items not in positive set)
            neg_idx = rng.choice(num_items)
            while neg_idx in pos_items:
                neg_idx = rng.choice(num_items)
            # non_pos_items = [i for i in item_range if i not in pos_items]
            # neg_idx = non_pos_items[rng.randint(0, len(non_pos_items))]

        valid_user_indices.append(user_idx)
        pos_item_indices.append(pos_idx)
        neg_item_indices.append(neg_idx)

    return valid_user_indices, pos_item_indices, neg_item_indices


def load_json_file(filepath):
    """
    Loads and parses a JSON file from the given filepath

    Args:
        filepath (str): Path to the JSON file

    Returns:
        dict: Parsed JSON data as a Python dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file isn't valid JSON
    """
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
    """Split a sparse COO matrix into train and test sets.

    Ensures row/column representation in training set for matrix factorization.

    Args:
        matrix: Sparse COO matrix to split
        train_ratio: Proportion for training set (default: 0.8)
        random_state: Random seed or RandomState (default: None)
        show_progress: Whether to display progress bars (default: False)

    Returns:
        (train_matrix, test_matrix) as COO matrices

    Raises:
        ValueError: If train_ratio not between 0 and 1
        TypeError: If input not a COO matrix
    """
    if not isinstance(matrix, sparse.coo_matrix):
        raise TypeError("Input matrix must be a scipy.sparse.coo_matrix")

    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1 inclusive")

    # Handle edge cases for train_ratio
    if train_ratio == 0:
        # All data goes to test set
        return sparse.coo_matrix(matrix.shape), matrix

    if train_ratio == 1:
        # All data goes to training set
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

    # Check if all rows and columns are represented in the training set
    train_rows = set(rows[train_idx])
    train_cols = set(cols[train_idx])

    # Find missing rows and columns in training set
    missing_rows = set(range(n_rows)) - train_rows
    missing_cols = set(range(n_cols)) - train_cols

    # For each missing row, find an entry in the test set and move it to train
    if show_progress and missing_rows:
        missing_row_iter = tqdm(missing_rows, desc="Train/Test Split (Rows)")
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

    # For each missing column, find an entry in the test set and move it to train
    if show_progress and missing_cols:
        missing_col_iter = tqdm(missing_cols, desc="Train/Test Split (Cols)")
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
    """ Get basic statistics about a sparse matrix."""
    stats = {
        "shape": matrix.shape,
        "nnz": matrix.nnz,
        "density": matrix.nnz / max((matrix.shape[0] * matrix.shape[1]), 1),
        "empty_rows": matrix.shape[0]-len(np.unique(matrix.row)),
        "empty_cols": matrix.shape[1]-len(np.unique(matrix.col)),
    }
    return stats


def print_sparse_matrix_stats(matrix: sparse.coo_matrix) -> None:
    """Prints compact stats about a sparse matrix in a single line."""
    stats = get_sparse_matrix_stats(matrix)
    print_str = (
        f"({stats['shape'][0]:6}×{stats['shape'][1]:6}) nnz={stats['nnz']:10,} "
        f"({stats['density']:5.3%}), "
        f"empty rows/cols={stats['empty_rows']:6}/{stats['empty_cols']:6}"
    )
    return print_str


# def random_col_for_row(
#     matrix: sparse.spmatrix,
#     row_idx: Union[int, Sequence[int]]
# ) -> Union[int, np.ndarray]:
#     """Get a random column for row(s) in a sparse matrix.

#     Args:
#         matrix: Any scipy.sparse matrix type
#         row_idx: The row index or sequence of row indices to find random columns for

#     Returns:
#         A random column index or numpy array of random column indices. For each row,
#         if it has non-zero elements, returns a random column where a non-zero
#         element exists. Otherwise, returns a random column from the entire matrix.
#     """
#     # --- Input validation and preprocessing ---
#     if not isinstance(matrix, sparse.spmatrix):
#         raise TypeError("Input must be a scipy.sparse matrix")

#     # Convert to CSR if needed
#     if not isinstance(matrix, sparse.csr_matrix):
#         csr_matrix = matrix.tocsr()
#     else:
#         csr_matrix = matrix

#     # Convert scalar to array if needed
#     is_scalar = isinstance(row_idx, int)
#     row_indices = np.array([row_idx]) if is_scalar else np.asarray(row_idx)

#     # # Validate row indices
#     # if np.any((row_indices < 0) | (row_indices >= csr_matrix.shape[0])):
#     #     raise ValueError(f"Row indices out of bounds for matrix with "
#     #                      f"{csr_matrix.shape[0]} rows")

#     # --- Get row data ---
#     # Get start and end indices for each row
#     starts = csr_matrix.indptr[row_indices]
#     ends = csr_matrix.indptr[row_indices + 1]

#     # Find rows with non-zero elements
#     non_empty_mask = starts < ends
#     non_empty_indices = np.where(non_empty_mask)[0]
#     empty_indices = np.where(~non_empty_mask)[0]
#     results = np.zeros(len(row_indices), dtype=np.int64)

#     # --- Process empty rows ---
#     if len(empty_indices) > 0:
#         results[empty_indices] = np.random.randint(
#             0, csr_matrix.shape[1], size=len(empty_indices))

#     # --- Process non-empty rows ---
#     if len(non_empty_indices) > 0:
#         # Calculate random positions for each non-empty row
#         row_lengths = ends[non_empty_indices] - starts[non_empty_indices]
#         random_offsets = np.random.random(len(non_empty_indices)) * row_lengths
#         random_offsets = np.floor(random_offsets).astype(np.int64)
#         positions = starts[non_empty_indices] + random_offsets

#         # Get the column indices
#         results[non_empty_indices] = csr_matrix.indices[positions]

#     # Return scalar or array based on input type
#     return results[0] if is_scalar else results


def series_to_categorical_int(
    series: pd.Series
) -> pd.Series:
    """
    Converts a pandas Series to categorical integers.

    Args:
        series: The input Series to convert.

    Returns:
        A Series of integers representing the categorical codes.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    return series.astype('category').cat.codes


def get_category_indices(
    series1: pd.Series,
    series2: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Aligns categories between two pandas Series.

    Args:
        series1: First series.
        series2: Second series.

    Returns:
        A tuple containing two Series of category indices.
    """
    # Get union of all unique values from both series
    all_categories = pd.Index(
        sorted(set(series1.dropna().unique()) | set(series2.dropna().unique()))
    )

    # Convert both series to categorical with the same categories
    cat_series1 = pd.Categorical(series1, categories=all_categories)
    cat_series2 = pd.Categorical(series2, categories=all_categories)

    # Get codes (indices) for each categorical series
    codes_series1 = pd.Series(cat_series1.codes, index=series1.index)
    codes_series2 = pd.Series(cat_series2.codes, index=series2.index)

    return codes_series1, codes_series2


def scipy_csr_to_torch_csr(scipy_csr):
    """
    Convert a SciPy sparse CSR matrix to a PyTorch sparse CSR tensor.

    Parameters:
    -----------
    scipy_csr : scipy.sparse.csr_matrix
        The SciPy CSR matrix to be converted.

    Returns:
    --------
    torch.Tensor
        A PyTorch sparse CSR tensor.
    """
    # Get the CSR components from the SciPy matrix
    data = scipy_csr.data
    indices = scipy_csr.indices
    indptr = scipy_csr.indptr
    shape = scipy_csr.shape

    # Convert numpy arrays to torch tensors with appropriate data types
    # Values keep their original dtype (typically float32 or float64)
    data_torch = torch.from_numpy(data)

    # Ensure indices and indptr are int64 (torch.int64)
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
    """Slice a SciPy sparse matrix by row indices and convert to PyTorch sparse.

    Args:
        scipy_matrix: Input SciPy sparse matrix to slice
        row_indices: Row indices to select from the matrix

    Returns:
        PyTorch sparse tensor containing the selected rows
    """
    # Convert to CSR for efficient row slicing if needed
    if not sp.isspmatrix_csr(scipy_matrix):
        scipy_matrix = scipy_matrix.tocsr()

    # Slice the matrix by row indices
    sliced_matrix = scipy_matrix[row_indices, :]

    # Convert to COO format (coordinate format)
    coo_matrix = sliced_matrix.tocoo()

    # Extract coordinates and values
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = torch.FloatTensor(coo_matrix.data)
    shape = torch.Size(coo_matrix.shape)

    # Create PyTorch sparse tensor
    return torch.sparse_coo_tensor(indices, values, shape)


# def csr_scipy_to_torch(csr_mat: sparray):
#     """Convert between csr sparse array from scipy to pytorch"""
#     return torch.sparse_csr_tensor(
#         torch.LongTensor(csr_mat.indptr),
#         torch.LongTensor(csr_mat.indices),
#         torch.FloatTensor(csr_mat.data),
#         size=torch.Size(csr_mat.shape)
#     )


# def get_indices_sorted_by_activity(
#     uimat: spmatrix,
#     axis: int,  # 0=row, 1=column
#     count: int | None = None
# ):
#     """return the most -count- active users (axis=0) or items"""
#     # get number of interactions for each user
#     count_vector = np.asarray(uimat.sum(axis=axis)).reshape(-1)
#     # get the index of users with most interactions at 0
#     sorted_list = count_vector.argsort()[::-1]
#     if count is not None:
#         sorted_list = sorted_list[:count]
#     return sorted_list


# def generate_user_item_indices(
#     parent_df: pd.DataFrame,
#     children_dfs: list[pd.DataFrame] = None,
#     userid_column: str = 'user_id',
#     itemid_column: str = 'item_id',
#     index_suffix: str = 'x'

# ):
    # """Function for appending user item indices to dataframe"""
    # for cname in [userid_column, itemid_column]:
    #     parent_df[cname] = parent_df[cname].astype('category')
    #     idx_cname = f'{cname}{index_suffix}'
    #     parent_df[idx_cname] = parent_df[cname].cat.codes.astype(int)
    #     if children_dfs is not None:
    #         for idf in children_dfs:
    #             if cname in idf.columns:
    #                 idf[cname] = pd.Categorical(
    #                     idf[cname],
    #                     categories=parent_df[cname].unique(),
    #                     ordered=False
    #                 )
    #                 idf[idx_cname] = idf[cname].cat.codes.astype(int)
    # if children_dfs is None:
    #     return parent_df

    # return parent_df, children_dfs


# def sample_pos_neg_pairs_old(
#     pos_csr_mat: csr_matrix,
#     neg_csr_mat: csr_matrix,
#     user_indices: Optional[List[int]] = None,
#     explicit_only: bool = False
# ) -> tuple[list[int], list[int], list[int]]:
#     """Sample one random non-zero entry from each row using LIL format"""
#     # Convert to LIL format for efficient row access
#     assert pos_csr_mat.shape == neg_csr_mat.shape, 'Shape mismatch'
#     pos_lil_mat = pos_csr_mat.tolil()
#     neg_lil_mat = neg_csr_mat.tolil()
#     set_of_all_items = set(range(pos_csr_mat.shape[1]))

#     # Initialize result lists
#     user_indices = user_indices if user_indices is not None else list(
#         range(pos_lil_mat.shape[0]))
#     valid_user_indices: List[int] = []
#     pos_item_indices: List[int] = []
#     neg_item_indices: List[int] = []

#     # Iterate through each row
#     for user_idx in user_indices:
#         plst = pos_lil_mat.rows[user_idx]
#         nlst = neg_lil_mat.rows[user_idx]
#         neg_idx = None
#         if len(plst) > 0:
#             pos_idx = np.random.choice(plst)
#             if len(nlst) > 0:
#                 neg_idx = np.random.choice(nlst)
#             else:
#                 if not explicit_only:
#                     non_pos_items = list(set_of_all_items - set(plst))
#                     neg_idx = np.random.choice(non_pos_items)
#             if neg_idx is not None:
#                 valid_user_indices.append(user_idx)
#                 pos_item_indices.append(pos_idx)
#                 neg_item_indices.append(neg_idx)

#     return valid_user_indices, pos_item_indices, neg_item_indices

# def get_cdf(data, **kwargs):
#     """returns cdf"""
#     count, bins_count = np.histogram(data, **kwargs)
#     pdf = count / sum(count)
#     cdf = np.cumsum(pdf)
#     return bins_count[1:], cdf


# def get_most_active_users_from_uimat(uimat, count: int | None = None):
#     """return the most -count- active users"""
#     # get number of interactions for each user
#     count_vector = np.asarray(uimat.sum(axis=1)).reshape(-1)
#     # get the index of users with most interactions at 0
#     sorted_user_list = count_vector.argsort()[::-1]
#     if count is not None:
#         sorted_user_list = sorted_user_list[:count]
#     return sorted_user_list


# def compute_metric(
#         umat: ndarray,  # user matrix
#         imat: ndarray,  # item matrix
#         uimat: spmatrix,  # user item sparse matrix
#         user_count: int,  # number of top users
#         num_recs: int
# ):
#     """"Compute the metric"""
#     num_users, num_items = uimat.shape
#     active_users = get_most_active_users_from_uimat(uimat, user_count)
#     umat_sliced = umat.take(active_users, axis=0)
#     rec_mat = umat_sliced.dot(imat.T)
#     top_recs = np.argsort(rec_mat)[:, ::-1][:, :num_recs]
#     rec_user_inds = np.repeat(active_users, num_recs)
#     rec_item_inds = np.ravel(top_recs)
#     rec_R = csr_matrix(
    #     (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
    #     dtype=np.int8,
    #     shape=(num_users, num_items)
    # )
    # rinds, _, _ = sp.find(rec_R + uimat)
    # # rinds, _, _ = sp.find(rec_R - rec_mat)
    # # n_rec = rec_R[self.most_active_users].count_nonzero()
    # # n_int = rec_mat[self.most_active_users].count_nonzero()
    # # ndcg_vec = np.frombuffer(ndcg_shm.buf)
#     # # ndcg_vec = (n_rec + n_int - rinds.size)/n_user
#     # return user_count*num_recs-(len(rinds)-uimat.nnz)


# # def update_shm(
# #     ituple, user_shm, item_shm,
# #     n_user, n_item, n_feat, rlambda,
# #     lrate, active_users, ndcg_shm, t_mat
# # ):
# #     """Update user and item matrix"""
# #     user_mat = np.frombuffer(user_shm.buf).reshape(
# #         n_user, n_feat)  # only need a specific row
# #     item_mat = np.frombuffer(item_shm.buf).reshape(
# #         n_item, n_feat)  # only need a specific row

# #     ith, user_uth, item_ith, item_jth = ituple
# #     user_u = user_mat[user_uth]
# #     item_i = item_mat[item_ith]
# #     item_j = item_mat[item_jth]
# #     r_uij = np.dot(user_u, (item_i - item_j))
# #     # compute this along with ndcg
# #     # r_uij = np.sum(user_u * (item_i - item_j), axis=1)
# #     # sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
# #     sigmoid = expit(r_uij)
# #     sigmoid_tiled = np.tile(sigmoid, (n_feat,))
# #     grad_u = np.multiply(sigmoid_tiled, (item_j - item_i))
# #     grad_u += rlambda * user_u
# #     grad_i = np.multiply(sigmoid_tiled, -user_u) + rlambda * item_i
# #     grad_j = np.multiply(sigmoid_tiled, user_u) + rlambda * item_j
# #     user_mat[user_uth] -= lrate * grad_u
# #     item_mat[item_ith] -= lrate * grad_i
# #     item_mat[item_jth] -= lrate * grad_j

# #     if ith == 10000:
# #         print(ith, flush=True)
# #         rec_mat = user_mat.take(active_users, axis=0).dot(item_mat.T)
# #         top_recs = np.argsort(rec_mat)[:, ::-1][:, :60]
# #         rec_user_inds = np.repeat(active_users, 60)
# #         rec_item_inds = np.ravel(top_recs)
# #         rec_R = csr_matrix(
# #             (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
# #             dtype=np.int8,
# #             shape=(n_user, n_item)
# #         )
# #         rinds, _, _ = sp.find(rec_R[active_users] - t_mat[active_users])
# #         n_rec = rec_R[active_users].count_nonzero()
# #         n_int = t_mat[active_users].count_nonzero()
# #         ndcg_vec = np.frombuffer(ndcg_shm.buf)
# #         ndcg_vec[ith] = (n_rec + n_int - rinds.size)/n_user

# #     #     ndcg_vec
# #     #     # batch dot product
# #     #     print(mp.current_process().name, len(active_users))
# #     # rec_mat = user_mat.take(active_users, axis=0).dot(
# #     #     item_mat.T)  # 5000*num_items
# #     # top_recs = np.argsort(rec_mat)[:, ::-1][:, :60] # 5000*60
# #     # rec_user_inds = np.repeat(active_users, 60)
# #     # rec_item_inds = np.ravel(top_recs)
# #     # rec_R = csr_matrix(
# #     #     (np.ones((len(rec_user_inds),)), (rec_user_inds, rec_item_inds)),
# #     #     dtype=np.int8,
# #     #     shape=(n_user, n_item)
# #     # )
# #     # rinds, _, _ = sp.find(rec_R[active_users] - t_mat[active_users])
# #     # n_rec = rec_R[active_users].count_nonzero()
# #     # n_int = t_mat[active_users].count_nonzero()
# #     # out = (n_rec + n_int - rinds.size)/n_user

# # def break_sparse_matrix(matrix: spmatrix, threshold: float):
# #     """
# #     Breaks a scipy sparse matrix into two sparse matrices based on a threshold value.

# #     Args:
# #         matrix: The input sparse matrix.
# #         threshold: The value used to split the matrix.

# #     Returns:
# #         Two sparse matrices: one with values below the threshold, and one with values above or equal to the threshold.
# #     """

# #     # Get the non-zero values and their indices
# #     assert issparse(matrix), 'Need scipy sparse matrix'
# #     row_indices, col_indices, data = find(matrix)

# #     # Create masks for values below and above/equal to the threshold
# #     below_mask = data < threshold
# #     above_mask = data >= threshold

# #     # Create new sparse matrices based on the masks
# #     below_matrix = csr_matrix(
# #         (data[below_mask], (row_indices[below_mask], col_indices[below_mask])),
# #         shape=matrix.shape
# #     )
# #     above_matrix = csr_matrix(
# #         (data[above_mask], (row_indices[above_mask], col_indices[above_mask])),
# #         shape=matrix.shape
# #     )

# #     return above_matrix, below_matrix


# # def get_perc_sparsity(smat: spmatrix) -> float:
# #     """Get the sparsity"""
# #     assert isinstance(smat, spmatrix), 'Need scipy sparse matrix as input!'
# #     nrow, ncol = smat.shape
# #     out = smat.nnz * 100 / (nrow * ncol)
# #     return out


# # def create_shared_memory_nparray(
# #     data: ndarray,
#     name: str,
#     dtype=np.float64
# ):
#     """Create shared memory object"""
#     d_size = np.dtype(dtype).itemsize * np.prod(data.shape)
#     shm = None
#     try:
#         shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
#     except FileExistsError:
#         release_shared(name)
#         shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
#     finally:
#         dst = np.ndarray(shape=data.shape, dtype=dtype, buffer=shm.buf)
#         dst[:] = data[:]
#     return shm


# def release_shared(name):
#     """Release shared memory block"""
#     shm = shared_memory.SharedMemory(name=name)
#     shm.close()
#     shm.unlink()  # Free and release the shared memory block


# def compute_mse(y_true, y_pred):
#     """ignore zero terms prior to comparing the mse"""
#     mask = np.nonzero(y_true)
#     assert mask[0].shape[0] > 0, 'Truth matrix empty'
#     mse = mean_squared_error(np.array(y_true[mask]).ravel(), y_pred[mask])
#     return mse


# def load_movielens_data(data_dir, flag='ml-100k'):
#     """Function to read movielens data"""
#     names = ['user_id', 'item_id', 'rating', 'timestamp']
#     if flag == 'ml-100k':
#         file_path = os.path.join(data_dir, flag, 'u.data')
#         df = pd.read_csv(file_path, sep='\t')
#         return df
#     elif flag == 'ml-1m':
#         file_path = os.path.join(data_dir, flag, 'ratings.dat')
#         df = pd.read_csv(file_path, sep='\t', engine='python')
#         return df
#     elif flag == 'ml-10M100K':
#         file_path = os.path.join(data_dir, flag, 'ratings.dat')
#         df = pd.read_csv(file_path, sep='::', names=names, engine='python')
#         return df
#     else:
#         raise ValueError('Choose among ml-100k, ml-1m, ml-10M100K')


# def compute_ndcg(
#     ranked_item_idx: List[int],
#     K: int,
#     wgt_fun: Callable = np.log2
# ):
#     """Comutes NDCG metric"""
#     assert K > 0, 'Should have atleast one recomendation, choose K >1'
#     ndcg_score = 0.
#     if np.array(ranked_item_idx).size > 0:
#         assert np.max(
#             ranked_item_idx) < K, 'entry in ranked_item_idx > K-1!'
#         Rup = np.zeros(K, dtype=int)
#         Rup[ranked_item_idx] = 1.
#         wgt = np.array([1 / wgt_fun(ix + 1) for ix in np.arange(1, K + 1)])
#         # wgt = np.array([1. for ix in np.arange(1, K + 1)])
#         ndcg_score = np.sum(np.multiply(wgt, Rup)) / np.sum(wgt)
#     return ndcg_score


# def get_interaction_weights(
#     train_mat,
#     strategy: str = 'same',
#     fac: float | None = None
# ):
#     """Function for getting the weights"""
#     row_inds, col_inds, _ = ss.find(train_mat)
#     num_users, num_items = train_mat.shape
#     match strategy.lower():
#         case 'positive-only':
#             weight_mat = np.zeros(train_mat.shape)
#             weight_mat[row_inds, col_inds] = 1.
#         case 'uniformly-negative':
#             weight_mat = np.random.uniform(size=train_mat.shape)
#             weight_mat[row_inds, col_inds] = 1.
#         case 'user-oriented':
#             fac = np.amax(train_mat.sum(axis=1)) if fac is None else fac
#             weight_mat = np.clip(train_mat.sum(axis=1), 0, fac) / fac
#             weight_mat = np.array(np.repeat(weight_mat, num_items, axis=1))
#             weight_mat[row_inds, col_inds] = 1.
#         case 'item-oriented':
#             fac = np.amax(train_mat.sum(axis=0)) if fac is None else fac
#             weight_mat = 1 - np.clip(train_mat.sum(axis=0), 0, fac) / fac
#             weight_mat = np.array(np.repeat(weight_mat, num_users, axis=0))
#             weight_mat[row_inds, col_inds] = 1.
#         case _:
#             weight_mat = np.ones(train_mat.shape)
#     return weight_mat


# def run_loop(func, tqdm_pbar: tqdm, ncores=mp.cpu_count()):
#     """Run parallel simulation"""
#     if ncores <= 1:
#         results = []
#         for ix in tqdm_pbar:
#             results.append(func(ix))
#     else:
#         with mp.Pool(ncores) as pool:
#             results = list(pool.imap(func, tqdm_pbar))
#     return results
