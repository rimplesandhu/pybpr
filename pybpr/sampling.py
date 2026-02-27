"""Negative sampling strategies."""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional, Set


def get_excluded_items(
    user_idx: int,
    Rpos_all_csr: csr_matrix,
) -> Set[int]:
    """Get items to exclude for user (all positives)."""
    return set(Rpos_all_csr[user_idx].indices)


def sample_random_negatives(
    n_neg: int,
    n_items: int,
    excluded_items: Set[int],
) -> np.ndarray:
    """Sample random negatives excluding specified items."""
    # Get available items for sampling
    all_items = set(range(n_items))
    available = np.array(list(all_items - excluded_items))

    if len(available) == 0:
        return np.array([], dtype=np.int32)

    # Sample without replacement
    n_neg = min(n_neg, len(available))
    return np.random.choice(available, size=n_neg, replace=False)


def sample_popular_negatives(
    n_neg: int,
    n_items: int,
    excluded_items: Set[int],
    item_popularity: np.ndarray,
) -> np.ndarray:
    """Sample negatives weighted by item popularity."""
    # Validate item_popularity length
    if len(item_popularity) != n_items:
        raise ValueError(
            f"item_popularity length {len(item_popularity)} "
            f"!= n_items {n_items}"
        )

    # Get available items for sampling
    all_items = set(range(n_items))
    available = np.array(list(all_items - excluded_items))

    if len(available) == 0:
        return np.array([], dtype=np.int32)

    n_neg = min(n_neg, len(available))
    pop_available = item_popularity[available]

    # Fall back to uniform if no popularity info
    if pop_available.sum() == 0:
        return np.random.choice(
            available, size=n_neg, replace=False
        )

    # Sample weighted by popularity
    probs = pop_available / pop_available.sum()
    return np.random.choice(
        available, size=n_neg, replace=False, p=probs
    )


def sample_similar_negatives(
    user_idx: int,
    n_neg: int,
    n_items: int,
    excluded_items: Set[int],
    Fi_csr: csr_matrix,
    Rpos_train_csr: csr_matrix,
) -> np.ndarray:
    """Sample negatives similar to user's positive items."""
    # Get available items for sampling
    all_items = set(range(n_items))
    available = np.array(list(all_items - excluded_items))

    if len(available) == 0:
        return np.array([], dtype=np.int32)

    # Get user's positive items for similarity computation
    pos_items = Rpos_train_csr[user_idx].indices
    if len(pos_items) == 0:
        n_neg = min(n_neg, len(available))
        return np.random.choice(
            available, size=n_neg, replace=False
        )

    # Compute average features and similarity scores
    pos_features = Fi_csr[pos_items].mean(axis=0)
    similarities = Fi_csr.dot(pos_features.T).toarray().flatten()
    similarities[list(excluded_items)] = -np.inf

    # Sample from top similar items
    top_k = min(n_neg * 10, len(similarities))
    candidates = np.argpartition(similarities, -top_k)[-top_k:]
    valid_candidates = np.intersect1d(candidates, available)

    if len(valid_candidates) == 0:
        return np.array([], dtype=np.int32)

    return np.random.choice(
        valid_candidates,
        size=min(n_neg, len(valid_candidates)),
        replace=False,
    )


def sample_model_based_negatives(
    n_neg: int,
    n_items: int,
    excluded_items: Set[int],
    scores: np.ndarray,
) -> np.ndarray:
    """Sample negatives with high model scores."""
    # Validate scores length
    if len(scores) != n_items:
        raise ValueError(
            f"scores length {len(scores)} != n_items {n_items}"
        )

    # Get available items for sampling
    all_items = set(range(n_items))
    available = np.array(list(all_items - excluded_items))

    if len(available) == 0:
        return np.array([], dtype=np.int32)

    # Mask excluded items
    scores_copy = scores.copy()
    scores_copy[list(excluded_items)] = -np.inf

    # Sample from high-scoring items
    top_k = min(n_neg * 10, len(scores_copy))
    candidates = np.argpartition(scores_copy, -top_k)[-top_k:]
    valid_candidates = np.intersect1d(candidates, available)

    if len(valid_candidates) == 0:
        return np.array([], dtype=np.int32)

    return np.random.choice(
        valid_candidates,
        size=min(n_neg, len(valid_candidates)),
        replace=False,
    )


def sample_stratified_negatives(
    n_neg: int,
    n_items: int,
    excluded_items: Set[int],
    item_popularity: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Sample mix of easy, medium, and hard negatives."""
    # Validate array lengths
    if len(item_popularity) != n_items:
        raise ValueError(
            f"item_popularity length {len(item_popularity)} "
            f"!= n_items {n_items}"
        )
    if len(scores) != n_items:
        raise ValueError(
            f"scores length {len(scores)} != n_items {n_items}"
        )

    # Get available items for sampling
    all_items = set(range(n_items))
    available = np.array(list(all_items - excluded_items))

    if len(available) == 0:
        return np.array([], dtype=np.int32)

    # Determine split: 20% easy, 30% medium, 50% hard
    n_easy = int(0.2 * n_neg)
    n_medium = int(0.3 * n_neg)
    n_hard = n_neg - n_easy - n_medium

    # Sample easy negatives (random)
    easy = np.random.choice(
        available, size=min(n_easy, len(available)), replace=False
    )
    available = np.setdiff1d(available, easy)

    if len(available) == 0:
        return easy

    # Sample medium negatives (popular items)
    pop_available = item_popularity[available]
    if pop_available.sum() > 0:
        probs = pop_available / pop_available.sum()
        medium = np.random.choice(
            available,
            size=min(n_medium, len(available)),
            replace=False,
            p=probs,
        )
    else:
        medium = np.random.choice(
            available,
            size=min(n_medium, len(available)),
            replace=False,
        )
    available = np.setdiff1d(available, medium)

    if len(available) == 0:
        return np.concatenate([easy, medium])

    # Sample hard negatives (high-scoring items)
    scores_available = scores[available]
    if len(scores_available) > 0:
        top_indices = np.argsort(scores_available)[
            -min(n_hard, len(available)):
        ]
        hard = available[top_indices]
    else:
        hard = np.array([], dtype=np.int32)

    return np.concatenate([easy, medium, hard])


def sample_negatives(
    user_idx: int,
    n_neg: int,
    strategy: str,
    n_items: int,
    Rpos_all_csr: csr_matrix,
    item_popularity: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    Fi_csr: Optional[csr_matrix] = None,
    Rpos_train_csr: Optional[csr_matrix] = None,
) -> np.ndarray:
    """Sample negatives using specified strategy."""
    # Get items to exclude for this user
    excluded = get_excluded_items(user_idx, Rpos_all_csr)

    # Random sampling
    if strategy == "random":
        return sample_random_negatives(n_neg, n_items, excluded)

    # Popularity-weighted sampling
    elif strategy == "popular":
        if item_popularity is None:
            raise ValueError("item_popularity required")
        return sample_popular_negatives(
            n_neg, n_items, excluded, item_popularity
        )

    # Similarity-based sampling
    elif strategy == "similar":
        if Fi_csr is None or Rpos_train_csr is None:
            raise ValueError("Fi_csr, Rpos_train_csr required")
        return sample_similar_negatives(
            user_idx, n_neg, n_items, excluded,
            Fi_csr, Rpos_train_csr
        )

    # Model-based sampling (hard negatives)
    elif strategy == "model_based":
        if scores is None:
            raise ValueError("scores required")
        return sample_model_based_negatives(
            n_neg, n_items, excluded, scores
        )

    # Stratified sampling (mixed difficulty)
    elif strategy == "stratified":
        if item_popularity is None or scores is None:
            raise ValueError("item_popularity and scores required")
        return sample_stratified_negatives(
            n_neg, n_items, excluded, item_popularity, scores
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
