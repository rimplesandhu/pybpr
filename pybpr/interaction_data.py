#!/usr/bin/env python3
"""User-Item interaction data management."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.sparse as sp

from .utils import print_sparse_matrix_stats


class UserItemData:
    """Manages user-item interaction and feature data."""

    def __init__(
        self,
        name: str,
        dtype: np.dtype = np.float32,
        verbose: bool = True
    ) -> None:
        """Initialize UserItemData with dynamic dimensions."""
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")

        if (not isinstance(dtype, type) or
                not issubclass(dtype, np.floating)):
            raise ValueError(
                "dtype must be a numpy floating point type"
            )

        self.name = name
        self.dtype = dtype
        self.verbose = verbose

        # Initialize dimensions
        self.n_users = 0
        self.n_items = 0
        self.n_user_features = 0
        self.n_item_features = 0

        # Initialize empty matrices
        self._Rpos = sp.coo_matrix((0, 0), dtype=dtype)
        self._Rneg = sp.coo_matrix((0, 0), dtype=dtype)
        self._Fu = sp.coo_matrix((0, 0), dtype=dtype)
        self._Fi = sp.coo_matrix((0, 0), dtype=dtype)

        # Train/test split matrices (None until split)
        self.Rpos_train = None
        self.Rpos_test = None
        self.Rneg_train = None
        self.Rneg_test = None

        # ID to index mappings: (id_to_idx, idx_to_id)
        self._id_to_idx_mappings = {
            'user': (dict(), dict()),
            'item': (dict(), dict()),
            'user_feature': (dict(), dict()),
            'item_feature': (dict(), dict())
        }

        if self.verbose:
            print(f"Initialized UserItemData '{name}' "
                  f"with dtype {dtype}")

    # def __repr__(self) -> str:
    #     """Return string representation of the dataset.

    #     Returns:
    #         Formatted string with dataset statistics
    #     """
    #     istr = (
    #         f"{self.__class__.__name__}({self.name})\n"
    #         f"  {'Fuser':10}:{print_sparse_matrix_stats(self.Fu)}\n"
    #         f"  {'Fitem':10}:{print_sparse_matrix_stats(self.Fi)}\n"
    #         f"  {'Rpos':10}:{print_sparse_matrix_stats(self.Rpos)}\n"
    #         f"  {'Rneg':10}:{print_sparse_matrix_stats(self.Rneg)}\n"
    #     )

    #     if self.Rpos_train is not None:
    #         istr += (
    #             f"  {'Rpos_train':10}:"
    #             f"{print_sparse_matrix_stats(self.Rpos_train)}\n"
    #             f"  {'Rpos_test':10}:"
    #             f"{print_sparse_matrix_stats(self.Rpos_test)}\n"
    #             f"  {'Rneg_train':10}:"
    #             f"{print_sparse_matrix_stats(self.Rneg_train)}\n"
    #             f"  {'Rneg_test':10}:"
    #             f"{print_sparse_matrix_stats(self.Rneg_test)}"
    #         )
    #     return istr

    def _get_index(
        self,
        input_id: int,
        mapping_type: str
    ) -> int:
        """Get or create index for given ID.

        Args:
            input_id: Original ID to map
            mapping_type: Type of mapping ('user', 'item', etc.)

        Returns:
            Internal index for the ID
        """
        if mapping_type not in self._id_to_idx_mappings:
            raise ValueError(f"Unknown mapping type: {mapping_type}")

        id_to_idx, idx_to_id = self._id_to_idx_mappings[mapping_type]

        if input_id not in id_to_idx:
            idx = len(id_to_idx)
            id_to_idx[input_id] = idx
            idx_to_id[idx] = input_id

            # Update dimensions dynamically
            if mapping_type == 'user':
                self.n_users = max(self.n_users, idx + 1)
            elif mapping_type == 'item':
                self.n_items = max(self.n_items, idx + 1)
            elif mapping_type == 'user_feature':
                self.n_user_features = max(self.n_user_features, idx + 1)
            elif mapping_type == 'item_feature':
                self.n_item_features = max(self.n_item_features, idx + 1)


        return id_to_idx[input_id]

    def _get_indices(
        self,
        ids: List[int],
        mapping_type: str
    ) -> List[int]:
        """Convert list of IDs to internal indices.

        Args:
            ids: List of original IDs
            mapping_type: Type of mapping

        Returns:
            List of internal indices
        """
        return [self._get_index(input_id, mapping_type) for input_id in ids]

    def get_id(self, idx: int, mapping_type: str) -> int:
        """Get original ID from internal index.

        Args:
            idx: Internal index
            mapping_type: Type of mapping

        Returns:
            Original ID
        """
        if mapping_type not in self._id_to_idx_mappings:
            raise ValueError(f"Unknown mapping type: {mapping_type}")

        _, idx_to_id = self._id_to_idx_mappings[mapping_type]
        if idx not in idx_to_id:
            raise ValueError(
                f"{mapping_type.capitalize()} index {idx} not found in mapping"
            )
        return idx_to_id[idx]

    def _process_weights(
        self,
        weights: Optional[Union[float, List[float]]],
        length: int
    ) -> np.ndarray:
        """Process and validate weights for interactions or features.

        Args:
            weights: Input weights (scalar, list, or None)
            length: Expected length of weights

        Returns:
            Processed weight array
        """
        if weights is None:
            return np.ones(length, dtype=self.dtype)

        if np.isscalar(weights):
            if not np.isfinite(weights):
                raise ValueError("Weight must be finite")
            return np.full(length, weights, dtype=self.dtype)

        if len(weights) != length:
            raise ValueError(
                f"Weight length ({len(weights)}) must match input length "
                f"({length})"
            )

        weight_array = np.array(weights, dtype=self.dtype)
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("All weights must be finite")

        return weight_array

    def _update_matrix(
        self,
        old_matrix: sp.coo_matrix,
        new_shape: Tuple[int, int],
        new_matrix: Optional[sp.coo_matrix] = None,
    ) -> sp.coo_matrix:
        """Update and resize sparse matrices.

        Args:
            old_matrix: Existing matrix to update
            new_shape: Target shape for the result
            new_matrix: New matrix to add (optional)

        Returns:
            Updated sparse matrix
        """
        # Validate shape
        if any(dim < 0 for dim in new_shape):
            raise ValueError("Matrix dimensions must be non-negative")

        if (new_matrix is not None and
                new_shape != new_matrix.shape):
            raise ValueError(
                f"Shape mismatch: expected {new_shape}, "
                f"got {new_matrix.shape}"
            )

        # Return appropriately shaped new matrix if old matrix is empty
        if old_matrix.nnz == 0:
            if new_matrix is not None:
                return new_matrix
            else:
                return sp.coo_matrix(new_shape, dtype=self.dtype)

        # Resize existing matrix if needed
        if (old_matrix.shape[0] < new_shape[0] or
                old_matrix.shape[1] < new_shape[1]):
            old_matrix = sp.coo_matrix(
                (old_matrix.data, (old_matrix.row, old_matrix.col)),
                shape=new_shape,
                dtype=self.dtype
            )

        # Combine matrices
        result = old_matrix
        if new_matrix is not None:
            result = old_matrix + new_matrix
            result.eliminate_zeros()

        return result

    def _reshape_all_matrices(self) -> None:
        """Reshape all matrices according to current dimensions."""

        # Update interaction matrices
        self._Rpos = self._update_matrix(
            self._Rpos, (self.n_users, self.n_items)
        )
        self._Rneg = self._update_matrix(
            self._Rneg, (self.n_users, self.n_items)
        )

        # Update feature matrices
        self._Fi = self._update_matrix(
            self._Fi, (self.n_items, self.n_item_features)
        )
        self._Fu = self._update_matrix(
            self._Fu, (self.n_users, self.n_user_features)
        )

    def add_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None,
        is_positive: bool = True
    ) -> None:
        """Add interactions to the dataset.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            weights: Interaction weights (optional)
            is_positive: Whether interactions are positive or negative
        """
        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"User and item ID lists must have equal length: "
                f"{len(user_ids)} != {len(item_ids)}"
            )

        if len(user_ids) == 0:
            return

        interaction_type = "positive" if is_positive else "negative"
        if self.verbose:
            print(
                f"Adding {len(user_ids):,} {interaction_type} "
                f"interactions"
            )

        # Convert IDs to indices
        user_indices = self._get_indices(user_ids, 'user')
        item_indices = self._get_indices(item_ids, 'item')

        # Process weights and create matrix
        values = self._process_weights(weights, len(user_indices))
        interaction_matrix = sp.coo_matrix(
            (values, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items),
            dtype=self.dtype
        )
        interaction_matrix.eliminate_zeros()

        # Update interaction matrix
        target_matrix = self._Rpos if is_positive else self._Rneg
        target_matrix = self._update_matrix(
            old_matrix=target_matrix,
            new_matrix=interaction_matrix,
            new_shape=(self.n_users, self.n_items)
        )

        # Update corresponding attribute
        if is_positive:
            self._Rpos = target_matrix
        else:
            self._Rneg = target_matrix

        # Reshape matrices if needed
        self._reshape_all_matrices()

        if self.verbose:
            print(
                f"Successfully added {interaction_type} interactions. "
                f"New dimensions: {self.n_users} users × "
                f"{self.n_items} items"
            )

    def add_positive_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add positive interactions to the dataset.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            weights: Interaction weights (optional)
        """
        self.add_interactions(user_ids, item_ids, weights, is_positive=True)

    def add_negative_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add negative interactions to the dataset.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            weights: Interaction weights (optional)
        """
        self.add_interactions(user_ids, item_ids, weights, is_positive=False)

    def add_user_features(
        self,
        user_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add user features and update dimensions.

        Args:
            user_ids: List of user IDs
            feature_ids: List of feature IDs
            feature_weights: Feature weights (optional)
        """
        if len(user_ids) != len(feature_ids):
            raise ValueError(
                f"User and feature ID lists must have equal length: "
                f"{len(user_ids)} != {len(feature_ids)}"
            )

        if len(user_ids) == 0:
            return

        if self.verbose:
            print(f"Adding {len(user_ids):,} user features")

        # Convert IDs to indices
        user_indices = self._get_indices(user_ids, 'user')
        feature_indices = self._get_indices(feature_ids, 'user_feature')

        # Process weights and create matrix
        values = self._process_weights(feature_weights, len(user_indices))
        feature_matrix = sp.coo_matrix(
            (values, (user_indices, feature_indices)),
            shape=(self.n_users, self.n_user_features),
            dtype=self.dtype
        )
        feature_matrix.eliminate_zeros()

        # Update user feature matrix
        self._Fu = self._update_matrix(
            old_matrix=self._Fu,
            new_matrix=feature_matrix,
            new_shape=(self.n_users, self.n_user_features)
        )
        self._reshape_all_matrices()

        if self.verbose:
            print(
                f"Added user features: {self.n_users} users × "
                f"{self.n_user_features} features"
            )

    def add_item_features(
        self,
        item_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add item features and update dimensions.

        Args:
            item_ids: List of item IDs
            feature_ids: List of feature IDs
            feature_weights: Feature weights (optional)
        """
        if len(item_ids) != len(feature_ids):
            raise ValueError(
                f"Item and feature ID lists must have equal length: "
                f"{len(item_ids)} != {len(feature_ids)}"
            )

        if len(item_ids) == 0:
            return

        if self.verbose:
            print(f"Adding {len(item_ids):,} item features")

        # Convert IDs to indices
        item_indices = self._get_indices(item_ids, 'item')
        feature_indices = self._get_indices(feature_ids, 'item_feature')

        # Process weights and create matrix
        values = self._process_weights(feature_weights, len(item_indices))
        feature_matrix = sp.coo_matrix(
            (values, (item_indices, feature_indices)),
            shape=(self.n_items, self.n_item_features),
            dtype=self.dtype
        )
        feature_matrix.eliminate_zeros()

        # Update item feature matrix
        self._Fi = self._update_matrix(
            old_matrix=self._Fi,
            new_matrix=feature_matrix,
            new_shape=(self.n_items, self.n_item_features)
        )
        self._reshape_all_matrices()

        if self.verbose:
            print(
                f"Added item features: {self.n_items} items × "
                f"{self.n_item_features} features"
            )

    def validate_dataset(self) -> None:
        """Validate the entire dataset."""
        if self.verbose:
            print("Validating dataset...")

        # Check for required data
        if self.Rpos.nnz == 0:
            raise ValueError("No positive interaction data found")

        # Check matrix dimensions consistency
        matrices_to_check = [
            ('Rpos', self.Rpos, (self.n_users, self.n_items)),
            ('Rneg', self.Rneg, (self.n_users, self.n_items)),
            ('Fu', self.Fu, (self.n_users, self.n_user_features)),
            ('Fi', self.Fi, (self.n_items, self.n_item_features))
        ]

        for name, matrix, expected_shape in matrices_to_check:
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"Matrix {name} has shape {matrix.shape}, "
                    f"expected {expected_shape}"
                )

        # Check feature coverage - warn if not all users/items have features
        if self._Fu.nnz > 0:
            users_with_features = set(self._Fu.row)
            if len(users_with_features) < self.n_users:
                missing_users = self.n_users - len(users_with_features)
                print("WARNING:", 
                    f"{missing_users}/{self.n_users} users have no features"
                )
        else:
            print("WARNING:", "No user features found")

        if self._Fi.nnz > 0:
            items_with_features = set(self._Fi.row)
            if len(items_with_features) < self.n_items:
                missing_items = self.n_items - len(items_with_features)
                print("WARNING:", 
                    f"{missing_items}/{self.n_items} items have no features"
                )
        else:
            print("WARNING: No item features found")

        if self.verbose:
            print("Dataset validation completed successfully")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save instance to file using joblib."""
        filepath = Path(filepath)
        if self.verbose:
            print(f"Saving dataset to {filepath}")

        try:
            # Create parent directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self, filepath, compress=3)
            file_size = filepath.stat().st_size / (1024 * 1024)
            if self.verbose:
                print(f"Saved ({file_size:.1f} MB)")
        except Exception as e:
            print(f"ERROR: Failed to save dataset: {e}")
            raise IOError(f"Failed to save dataset: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'UserItemData':
        """Load instance from file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            instance = joblib.load(filepath)

            if not isinstance(instance, cls):
                raise TypeError(
                    f"Loaded object is not {cls.__name__}, "
                    f"got {type(instance).__name__}"
                )

            # Ensure verbose attribute exists
            if not hasattr(instance, 'verbose'):
                instance.verbose = True

            file_size = filepath.stat().st_size / (1024 * 1024)
            if instance.verbose:
                print(
                    f"Loaded dataset from {filepath} "
                    f"({file_size:.1f} MB)"
                )
            return instance

        except Exception as e:
            print(f"ERROR: Failed to load dataset from {filepath}: {e}")
            raise IOError(f"Failed to load dataset: {e}")

    def _get_interaction_stats(
        self,
        matrix: sp.coo_matrix
    ) -> str:
        """Get min/max interaction statistics for a matrix.

        Args:
            matrix: Sparse interaction matrix

        Returns:
            Formatted string with user/item interaction statistics
        """
        if matrix.nnz == 0:
            return ""

        # Convert to CSR for efficient row/column operations
        matrix_csr = matrix.tocsr()
        interactions_per_user = np.array(
            matrix_csr.sum(axis=1)
        ).flatten()
        interactions_per_item = np.array(
            matrix_csr.sum(axis=0)
        ).flatten()

        # Filter to non-zero counts
        user_counts = interactions_per_user[interactions_per_user > 0]
        item_counts = interactions_per_item[interactions_per_item > 0]

        stats = ""
        if len(user_counts) > 0:
            stats += (
                f"users: min={int(user_counts.min())}, "
                f"max={int(user_counts.max())} | "
            )
        if len(item_counts) > 0:
            stats += (
                f"items: min={int(item_counts.min())}, "
                f"max={int(item_counts.max())}"
            )

        return stats

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        istr = f"{self.__class__.__name__}({self.name})\n"
        istr += f"  Fuser: {print_sparse_matrix_stats(self.Fu)}\n"
        istr += f"  Fitem: {print_sparse_matrix_stats(self.Fi)}\n"
        istr += f"  Rpos:  {print_sparse_matrix_stats(self.Rpos)}\n"
        rpos_stats = self._get_interaction_stats(self.Rpos)
        if rpos_stats:
            istr += f"         └─ {rpos_stats}\n"
        istr += f"  Rneg:  {print_sparse_matrix_stats(self.Rneg)}"
        rneg_stats = self._get_interaction_stats(self.Rneg)
        if rneg_stats:
            istr += f"\n         └─ {rneg_stats}"
        return istr
    
    @property
    def user_ids_in_interactions(self) -> List[int]:
        """Get all user IDs that have interactions.

        Returns:
            Sorted list of user IDs
        """
        return sorted(self._id_to_idx_mappings['user'][0].keys())

    @property
    def item_ids_in_interactions(self) -> List[int]:
        """Get all item IDs that have interactions.

        Returns:
            Sorted list of item IDs
        """
        return sorted(self._id_to_idx_mappings['item'][0].keys())

    @property
    def Rpos(self) -> sp.coo_matrix:
        """Get positive interactions matrix."""
        return self._Rpos

    @property
    def Rneg(self) -> sp.coo_matrix:
        """Get negative interactions matrix."""
        return self._Rneg

    @property
    def Fu(self) -> sp.coo_matrix:
        """Get user features matrix."""
        return self._Fu

    @property
    def Fi(self) -> sp.coo_matrix:
        """Get item features matrix."""
        return self._Fi

    def split_train_test(
        self,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None
    ) -> None:
        """Split interactions into train and test sets.

        Args:
            train_ratio: Fraction of data for training
            random_state: Random seed for reproducibility
        """
        from .utils import split_sparse_coo_matrix

        # Split positive interactions
        self.Rpos_train, self.Rpos_test = (
            split_sparse_coo_matrix(
                self._Rpos, train_ratio, random_state
            )
        )

        # Split negative interactions
        if self._Rneg.nnz > 0:
            self.Rneg_train, self.Rneg_test = (
                split_sparse_coo_matrix(
                    self._Rneg, train_ratio, random_state
                )
            )
        else:
            # Empty negative matrices
            shape = self._Rpos.shape
            self.Rneg_train = sp.coo_matrix(
                shape, dtype=self.dtype
            )
            self.Rneg_test = sp.coo_matrix(
                shape, dtype=self.dtype
            )

        if self.verbose:
            print(f"Split data: train_ratio={train_ratio}")
