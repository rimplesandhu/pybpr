from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

from .utils import split_sparse_coo_matrix, print_sparse_matrix_stats


class UserItemData:
    """Manages user-item interaction and feature data for recommendation systems."""

    def __init__(
        self,
        name: str,
        dtype: np.dtype = np.float32
    ):
        """Initialize UserItemData object with dynamic dimensions."""
        self.name = name
        self.dtype = dtype

        # Initialize dimensions to zero - will be updated dynamically
        self.n_users = 0
        self.n_items = 0
        self.n_user_features = 0
        self.n_item_features = 0

        # Initialize empty matrices
        self._Rpos = sp.coo_matrix((0, 0), dtype=dtype)
        self._Rneg = sp.coo_matrix((0, 0), dtype=dtype)
        self._Rpos_train = None
        self._Rpos_test = None
        self._Rneg_train = None
        self._Rneg_test = None
        self._Fu = sp.coo_matrix((0, 0), dtype=dtype)
        self._Fi = sp.coo_matrix((0, 0), dtype=dtype)

        # ID to index mappings
        self._id_to_idx_mappings = {
            'user': (dict(), dict()),
            'item': (dict(), dict()),
            'user_feature': (dict(), dict()),
            'item_feature': (dict(), dict())
        }

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        istr = (
            f"{self.__class__.__name__}({self.name})\n"
            f"  {'Fuser':10}:{print_sparse_matrix_stats(self.Fu)}\n"
            f"  {'Fitem':10}:{print_sparse_matrix_stats(self.Fi)}\n"
            f"  {'Rpos':10}:{print_sparse_matrix_stats(self.Rpos)}\n"
            f"  {'Rneg':10}:{print_sparse_matrix_stats(self.Rneg)}\n"
        )
        if self.Rpos_train is not None:
            istr += (
                f"  {'Rpos_train':10}:{print_sparse_matrix_stats(self.Rpos_train)}\n"
                f"  {'Rpos_test':10}:{print_sparse_matrix_stats(self.Rpos_test)}\n"
                f"  {'Rneg_train':10}:{print_sparse_matrix_stats(self.Rneg_train)}\n"
                f"  {'Rneg_test':10}:{print_sparse_matrix_stats(self.Rneg_test)}"
            )
        return istr

    def _get_index(
        self,
        input_id: int,
        mapping_type: str
    ) -> int:
        """Get or create an index for a given ID and mapping type """
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
                self.n_user_features = max(
                    self.n_user_features, idx + 1
                )
            elif mapping_type == 'item_feature':
                self.n_item_features = max(
                    self.n_item_features, idx + 1
                )

        return id_to_idx[input_id]

    def _get_indices(
        self,
        ids: List[int],
        mapping_type: str
    ) -> List[int]:
        """ Convert a list of IDs to internal indices"""
        return [self._get_index(input_id, mapping_type)
                for input_id in ids]

    def get_id(self, idx: int, mapping_type: str) -> int:
        """ Get original ID from internal index"""
        _, idx_to_id = self._id_to_idx_mappings[mapping_type]
        if idx not in idx_to_id:
            raise ValueError(f"{mapping_type.capitalize()} index {idx} "
                             "not found in mapping")
        return idx_to_id[idx]

    def _process_weights(
        self,
        weights: Optional[Union[float, List[float]]],
        length: int
    ) -> np.ndarray:
        """Process and validate weights for interactions or features."""
        if weights is None:
            return np.ones(length, dtype=self.dtype)
        if np.isscalar(weights):
            return np.full(length, weights, dtype=self.dtype)
        if len(weights) != length:
            raise ValueError(
                f"Weight length ({len(weights)}) must match input length ({length})"
            )
        return np.array(weights, dtype=self.dtype)

    def _update_matrix(
        self,
        old_matrix: sp.coo_matrix,
        new_shape: Tuple[int, int],
        new_matrix: Optional[sp.coo_matrix] = None,
    ) -> sp.coo_matrix:
        """Helper method to update and resize sparse matrices."""
        # Validate shape
        if new_matrix is not None and new_shape != new_matrix.shape:
            raise ValueError('Shape mismatch between new_mat and new_shape')

        # return new mat if old mat is empty
        if old_matrix.nnz == 0 and new_matrix is not None:
            return new_matrix

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

    def _reshape_all_matrices(self):
        """Reshape according to the new data"""
        self._Rpos = self._update_matrix(
            self._Rpos, (self.n_users, self.n_items))
        self._Rneg = self._update_matrix(
            self._Rneg, (self.n_users, self.n_items))
        self._Fi = self._update_matrix(
            self._Fi, (self.n_items, self.n_item_features))
        self._Fu = self._update_matrix(
            self._Fu, (self.n_users, self.n_user_features))
        if self._Rpos_test is not None:
            self._Rpos_train = None
            self._Rpos_test = None
            self._Rneg_train = None
            self._Rneg_test = None
            print('New data added; Make sure to rerun train_test_split!')

    def add_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None,
        is_positive: bool = True
    ) -> None:
        """Add interactions to the dataset"""
        if len(user_ids) != len(item_ids):
            raise ValueError("User and item IDs must have equal length")

        if len(user_ids) == 0:
            return

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

    def add_positive_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add positive interactions to the dataset."""
        self.add_interactions(user_ids, item_ids, weights, is_positive=True)

    def add_negative_interactions(
        self,
        user_ids: List[int],
        item_ids: List[int],
        weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add negative interactions to the dataset."""
        self.add_interactions(user_ids, item_ids, weights, is_positive=False)

    def add_user_features(
        self,
        user_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add user features and update dimensions."""
        if len(user_ids) != len(feature_ids):
            raise ValueError("User and feature IDs must have equal length")

        if len(user_ids) == 0:
            return

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

    def add_item_features(
        self,
        item_ids: List[int],
        feature_ids: List[int],
        feature_weights: Optional[Union[float, List[float]]] = None
    ) -> None:
        """Add item features and update dimensions."""
        if len(item_ids) != len(feature_ids):
            raise ValueError("Item and feature IDs must have equal length")

        if len(item_ids) == 0:
            return

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

    def validate_dataset(self) -> None:
        """Validate the entire dataset."""
        # Check for required data
        if self.Rpos.nnz == 0:
            raise ValueError("No pos interaction data found")
        if self._Rpos_train is None:
            raise ValueError("Train-test split not performed")
        if self._Fu.nnz == 0:
            raise ValueError("No user feature data found")
        if self._Fi.nnz == 0:
            raise ValueError("No item feature data found")

        # # Check coverage
        # users_with_features = set(self._Fu.row)
        # items_with_features = set(self._Fi.row)

        # if len(users_with_features) < self.n_users:
        #     missing_users = self.n_users - len(users_with_features)
        #     raise ValueError(f"{missing_users} users have no feature data")
        # if len(items_with_features) < self.n_items:
        #     missing_items = self.n_items - len(items_with_features)
        #     raise ValueError(f"{missing_items} items have no feature data")

    def train_test_split(
        self,
        train_ratio_pos: float,
        train_ratio_neg: float,
        random_state: Optional[int] = None,
        show_progress: bool = True
    ) -> None:
        """Split positive and negative interactions into train/test sets."""

        # Split positive interactions
        self._Rpos_train, self._Rpos_test = split_sparse_coo_matrix(
            self.Rpos,
            train_ratio_pos,
            random_state,
            show_progress
        )

        # Split negative interactions if they exist
        self._Rneg_train, self._Rneg_test = split_sparse_coo_matrix(
            self.Rneg,
            train_ratio_neg,
            random_state,
            show_progress
        )

    @property
    def user_ids_in_interactions(self) -> List:
        """Give all user ids"""
        return list(self._id_to_idx_mappings['user'][0].keys())

    @property
    def item_ids_in_interactions(self) -> List:
        """Give all user ids"""
        return list(self._id_to_idx_mappings['item'][0].keys())

    @property
    def Rpos(self) -> sp.coo_matrix:
        """Get positive interactions matrix."""
        return self._Rpos

    @property
    def Rpos_train(self) -> sp.coo_matrix:
        """Get training positive interactions matrix."""
        return self._Rpos_train

    @property
    def Rpos_test(self) -> sp.coo_matrix:
        """Get testing positive interactions matrix."""
        return self._Rpos_test

    @property
    def Rneg(self) -> sp.coo_matrix:
        """Get negative interactions matrix."""
        return self._Rneg

    @property
    def Rneg_train(self) -> sp.coo_matrix:
        """Get training negative interactions matrix."""
        return self._Rneg_train

    @property
    def Rneg_test(self) -> sp.coo_matrix:
        """Get testing negative interactions matrix."""
        return self._Rneg_test

    @property
    def Fu(self) -> sp.coo_matrix:
        """Get user features matrix."""
        return self._Fu

    @property
    def Fi(self) -> sp.coo_matrix:
        """Get item features matrix."""
        return self._Fi


# """Base class for defining User-Item interaction data."""

# from typing import Dict, List, Optional, Tuple, Union

# import numpy as np
# import scipy.sparse as sp
# from tqdm import tqdm


# class UserItemData:
#     """Manages user-item interaction and feature data for recommendation systems."""

#     def __init__(
#         self,
#         name: str,
#         dtype: np.dtype = np.float32
#     ):
#         """Initialize UserItemData object with dynamic dimensions."""
#         self.name = name
#         self.dtype = dtype

#         # Initialize dimensions to zero - will be updated dynamically
#         self.n_users = 0
#         self.n_items = 0
#         self.n_user_features = 0
#         self.n_item_features = 0

#         # Initialize empty matrices
#         self._Rpos = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Rneg = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Rpos_train = None
#         self._Rpos_test = None
#         self._Fu = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Fi = sp.coo_matrix((0, 0), dtype=dtype)

#         # ID to index mappings
#         self._user_id_to_idx: Dict[int, int] = {}
#         self._item_id_to_idx: Dict[int, int] = {}
#         self._user_idx_to_id: Dict[int, int] = {}
#         self._item_idx_to_id: Dict[int, int] = {}
#         self._user_feature_id_to_idx: Dict[int, int] = {}
#         self._item_feature_id_to_idx: Dict[int, int] = {}
#         self._user_feature_idx_to_id: Dict[int, int] = {}
#         self._item_feature_idx_to_id: Dict[int, int] = {}

#     def __repr__(self) -> str:
#         """Return string representation of the dataset."""
#         n_total = self.Rpos.nnz + self.Rneg.nnz
#         pos_perc = (np.round(self.Rpos.nnz * 100 / n_total, 1)
#                     if n_total > 0 else 0)
#         return (f"{self.__class__.__name__}(\n"
#                 f"  name='{self.name}'\n"
#                 f"  n_users={self.n_users:,}\n"
#                 f"  n_items={self.n_items:,}\n"
#                 f"  n_interactions={n_total:,} ({pos_perc}% pos)\n"
#                 f"  n_user_features={self.n_user_features:,}"
#                 f" (n_data={self._Fu.nnz:,})\n"
#                 f"  n_item_features={self.n_item_features:,}"
#                 f" (n_data={self._Fi.nnz:,})\n)")

#     def _get_user_index(self, user_id: int) -> int:
#         """
#         Get the internal index for a user ID.
#         If the ID doesn't exist, create a new mapping.
#         """
#         if user_id not in self._user_id_to_idx:
#             idx = len(self._user_id_to_idx)
#             self._user_id_to_idx[user_id] = idx
#             self._user_idx_to_id[idx] = user_id
#             self.n_users = max(self.n_users, idx + 1)
#         return self._user_id_to_idx[user_id]

#     def _get_item_index(self, item_id: int) -> int:
#         """
#         Get the internal index for an item ID.
#         If the ID doesn't exist, create a new mapping.
#         """
#         if item_id not in self._item_id_to_idx:
#             idx = len(self._item_id_to_idx)
#             self._item_id_to_idx[item_id] = idx
#             self._item_idx_to_id[idx] = item_id
#             self.n_items = max(self.n_items, idx + 1)
#         return self._item_id_to_idx[item_id]

#     def _get_user_indices(self, user_ids: List[int]) -> List[int]:
#         """Convert a list of user IDs to internal indices."""
#         return [self._get_user_index(user_id) for user_id in user_ids]

#     def _get_item_indices(self, item_ids: List[int]) -> List[int]:
#         """Convert a list of item IDs to internal indices."""
#         return [self._get_item_index(item_id) for item_id in item_ids]

#     def _get_user_feature_index(self, feature_id: int) -> int:
#         """
#         Get the internal index for a user feature ID.
#         If the ID doesn't exist, create a new mapping.
#         """
#         if feature_id not in self._user_feature_id_to_idx:
#             idx = len(self._user_feature_id_to_idx)
#             self._user_feature_id_to_idx[feature_id] = idx
#             self._user_feature_idx_to_id[idx] = feature_id
#             self.n_user_features = max(self.n_user_features, idx + 1)
#         return self._user_feature_id_to_idx[feature_id]

#     def _get_item_feature_index(self, feature_id: int) -> int:
#         """
#         Get the internal index for an item feature ID.
#         If the ID doesn't exist, create a new mapping.
#         """
#         if feature_id not in self._item_feature_id_to_idx:
#             idx = len(self._item_feature_id_to_idx)
#             self._item_feature_id_to_idx[feature_id] = idx
#             self._item_feature_idx_to_id[idx] = feature_id
#             self.n_item_features = max(self.n_item_features, idx + 1)
#         return self._item_feature_id_to_idx[feature_id]

#     def _get_user_feature_indices(self, feature_ids: List[int]) -> List[int]:
#         """Convert a list of user feature IDs to internal indices."""
#         return [self._get_user_feature_index(fid) for fid in feature_ids]

#     def _get_item_feature_indices(self, feature_ids: List[int]) -> List[int]:
#         """Convert a list of item feature IDs to internal indices."""
#         return [self._get_item_feature_index(fid) for fid in feature_ids]

#     def get_user_id(self, idx: int) -> int:
#         """Get original user ID from internal index."""
#         if idx not in self._user_idx_to_id:
#             raise ValueError(f"User index {idx} not found in mapping")
#         return self._user_idx_to_id[idx]

#     def get_item_id(self, idx: int) -> int:
#         """Get original item ID from internal index."""
#         if idx not in self._item_idx_to_id:
#             raise ValueError(f"Item index {idx} not found in mapping")
#         return self._item_idx_to_id[idx]

#     def get_user_feature_id(self, idx: int) -> int:
#         """Get original user feature ID from internal index."""
#         if idx not in self._user_feature_idx_to_id:
#             raise ValueError(f"User feature index {idx} not found in mapping")
#         return self._user_feature_idx_to_id[idx]

#     def get_item_feature_id(self, idx: int) -> int:
#         """Get original item feature ID from internal index."""
#         if idx not in self._item_feature_idx_to_id:
#             raise ValueError(f"Item feature index {idx} not found in mapping")
#         return self._item_feature_idx_to_id[idx]

#     def train_test_split(
#         self,
#         test_ratio: float = 0.2,
#         random_state: Optional[int] = None,
#         show_progress: bool = True
#     ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
#         """Efficiently split interactions with all users/items in training set."""
#         if not 0 < test_ratio < 1:
#             raise ValueError("test_ratio must be between 0 and 1")
#         if random_state is not None:
#             np.random.seed(random_state)

#         # Initialize matrices and process each user
#         csr = self.Rpos.tocsr()
#         lil_train = sp.lil_matrix(csr.shape, dtype=self.dtype)
#         lil_test = sp.lil_matrix(csr.shape, dtype=self.dtype)
#         active_users = np.unique(csr.nonzero()[0])

#         # Process each user efficiently
#         for user in tqdm(active_users, desc="Splitting users",
#                          disable=not show_progress):
#             _, item_indices = csr[user].nonzero()
#             if len(item_indices) == 0:
#                 continue

#             # Split interactions
#             np.random.shuffle(item_indices)
#             n_test = min(max(int(len(item_indices) * test_ratio), 0),
#                          len(item_indices) - 1)
#             test_items = item_indices[:n_test]
#             train_items = item_indices[n_test:]

#             # Get values and populate matrices
#             train_values = np.array([csr[user, i] for i in train_items])
#             test_values = np.array([csr[user, i] for i in test_items])
#             for i, val in zip(train_items, train_values):
#                 lil_train[user, i] = val
#             for i, val in zip(test_items, test_values):
#                 lil_test[user, i] = val

#         # Ensure all items are in the training set
#         train_items_present = set(lil_train.nonzero()[1])
#         all_items = set(self.Rpos.col)
#         missing_items = all_items - train_items_present

#         if missing_items:
#             print(
#                 f"Found {len(missing_items)} items missing from training set")
#             test_coords = list(zip(*lil_test.nonzero()))
#             np.random.shuffle(test_coords)

#             for item in tqdm(missing_items, desc="Fixing missing items",
#                              disable=not show_progress):
#                 for user, test_item in test_coords:
#                     if test_item == item:
#                         # Move interaction to training
#                         lil_train[user, item] = lil_test[user, item]
#                         lil_test[user, item] = 0
#                         break

#         # Convert matrices and gather statistics
#         print("Converting matrices to COO format...")
#         self._Rpos_train = lil_train.tocoo()
#         self._Rpos_test = lil_test.tocoo()

#         # Calculate and print statistics
#         train_users, test_users = set(
#             self._Rpos_train.row), set(self._Rpos_test.row)
#         train_items, test_items = set(
#             self._Rpos_train.col), set(self._Rpos_test.col)

#         # Print basic statistics
#         print(f"{self.__class__.__name__}: Train-test split completed\n"
#               f"  Train interactions: {self._Rpos_train.nnz:,}\n"
#               f"  Test interactions: {self._Rpos_test.nnz:,}\n"
#               f"  Users in train/test: {len(train_users):,}/{len(test_users):,}\n"
#               f"  Items in train/test: {len(train_items):,}/{len(test_items):,}\n")

#         return self._Rpos_train, self._Rpos_test

#     def _process_weights(
#         self,
#         weights: Optional[Union[float, List[float]]],
#         length: int
#     ) -> np.ndarray:
#         """Process and validate weights."""
#         if weights is None:
#             return np.ones(length, dtype=self.dtype)
#         if np.isscalar(weights):
#             return np.full(length, weights, dtype=self.dtype)
#         if len(weights) != length:
#             raise ValueError(
#                 f"Weight length ({len(weights)}) must match input length ({length})"
#             )
#         return np.array(weights, dtype=self.dtype)

#     def _update_matrix(
#         self,
#         old_matrix: sp.coo_matrix,
#         new_data: sp.coo_matrix,
#         new_shape: Tuple[int, int]
#     ) -> sp.coo_matrix:
#         """Helper method to update and resize sparse matrices."""
#         if old_matrix.nnz == 0:
#             return new_data

#         # Resize existing matrix if needed
#         if (old_matrix.shape[0] < new_shape[0] or
#                 old_matrix.shape[1] < new_shape[1]):
#             old_matrix = sp.coo_matrix(
#                 (old_matrix.data, (old_matrix.row, old_matrix.col)),
#                 shape=new_shape,
#                 dtype=self.dtype
#             )

#         # Combine matrices
#         result = old_matrix + new_data
#         result.eliminate_zeros()
#         return result

#     def add_positive_interactions(
#         self,
#         user_ids: List[int],
#         item_ids: List[int],
#         weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add positive interactions to the dataset and update dimensions."""
#         if len(user_ids) != len(item_ids):
#             raise ValueError("User and item IDs must have equal length")

#         if len(user_ids) == 0:
#             return

#         # Convert IDs to indices
#         user_indices = self._get_user_indices(user_ids)
#         item_indices = self._get_item_indices(item_ids)

#         # Process weights and create matrix
#         values = self._process_weights(weights, len(user_indices))

#         interaction_matrix = sp.coo_matrix(
#             (values, (user_indices, item_indices)),
#             shape=(self.n_users, self.n_items),
#             dtype=self.dtype
#         )
#         interaction_matrix.eliminate_zeros()

#         # Update positive interaction matrix
#         self._Rpos = self._update_matrix(
#             self._Rpos,
#             interaction_matrix,
#             (self.n_users, self.n_items)
#         )

#         # Resize negative interaction matrix if necessary
#         if self._Rneg.nnz > 0 and (self._Rneg.shape[0] < self.n_users or
#                                    self._Rneg.shape[1] < self.n_items):
#             self._Rneg = sp.coo_matrix(
#                 (self._Rneg.data, (self._Rneg.row, self._Rneg.col)),
#                 shape=(self.n_users, self.n_items),
#                 dtype=self.dtype
#             )

#     def add_negative_interactions(
#         self,
#         user_ids: List[int],
#         item_ids: List[int],
#         weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """
#         Add negative interactions to the dataset.
#         Only users with existing positive interactions can have negative
#         interactions.
#         """
#         if len(user_ids) != len(item_ids):
#             raise ValueError("User and item IDs must have equal length")

#         if len(user_ids) == 0:
#             return

#         # Convert IDs to indices
#         user_indices = self._get_user_indices(user_ids)
#         item_indices = self._get_item_indices(item_ids)

#         # Get users with positive interactions
#         users_with_pos = set(self._Rpos.row)

#         # Filter out users without positive interactions
#         valid_indices = [(i, u, item) for i, (u, item) in
#                          enumerate(zip(user_indices, item_indices))
#                          if u in users_with_pos]

#         if len(valid_indices) < len(user_indices):
#             n_ignored = len(user_indices) - len(valid_indices)
#             print(f"Ignoring {n_ignored} negative interactions from users "
#                   f"with no positive interactions "
#                   f"({n_ignored/len(user_indices):.1%} of total)")

#             if not valid_indices:
#                 print("No valid negative interactions remaining, skipping")
#                 return

#             # Extract valid indices
#             valid_idx, user_indices, item_indices = zip(*valid_indices)
#             if weights is not None and not np.isscalar(weights):
#                 weights = [weights[i] for i in valid_idx]

#         # Process weights and create matrix
#         values = self._process_weights(weights, len(user_indices))

#         interaction_matrix = sp.coo_matrix(
#             (values, (user_indices, item_indices)),
#             shape=(self.n_users, self.n_items),
#             dtype=self.dtype
#         )
#         interaction_matrix.eliminate_zeros()

#         # Update negative interaction matrix
#         self._Rneg = self._update_matrix(
#             self._Rneg,
#             interaction_matrix,
#             (self.n_users, self.n_items)
#         )

#     def add_user_features(
#         self,
#         user_ids: List[int],
#         feature_ids: List[int],
#         feature_weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add user features and update dimensions."""
#         if len(user_ids) != len(feature_ids):
#             raise ValueError("User and feature IDs must have equal length")

#         if len(user_ids) == 0:
#             return

#         # Convert IDs to indices
#         user_indices = self._get_user_indices(user_ids)
#         feature_indices = self._get_user_feature_indices(feature_ids)

#         # Process weights and create matrix
#         values = self._process_weights(feature_weights, len(user_indices))

#         feature_matrix = sp.coo_matrix(
#             (values, (user_indices, feature_indices)),
#             shape=(self.n_users, self.n_user_features),
#             dtype=self.dtype
#         )
#         feature_matrix.eliminate_zeros()

#         # Update user feature matrix
#         self._Fu = self._update_matrix(
#             self._Fu,
#             feature_matrix,
#             (self.n_users, self.n_user_features)
#         )

#     def add_item_features(
#         self,
#         item_ids: List[int],
#         feature_ids: List[int],
#         feature_weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add item features and update dimensions."""
#         if len(item_ids) != len(feature_ids):
#             raise ValueError("Item and feature IDs must have equal length")

#         if len(item_ids) == 0:
#             return

#         # Convert IDs to indices
#         item_indices = self._get_item_indices(item_ids)
#         feature_indices = self._get_item_feature_indices(feature_ids)

#         # Process weights and create matrix
#         values = self._process_weights(feature_weights, len(item_indices))

#         feature_matrix = sp.coo_matrix(
#             (values, (item_indices, feature_indices)),
#             shape=(self.n_items, self.n_item_features),
#             dtype=self.dtype
#         )
#         feature_matrix.eliminate_zeros()

#         # Update item feature matrix
#         self._Fi = self._update_matrix(
#             self._Fi,
#             feature_matrix,
#             (self.n_items, self.n_item_features)
#         )

#     def validate_dataset(self) -> None:
#         """Validate the entire dataset."""
#         # Check for required data
#         if self.Rpos.nnz == 0:
#             raise ValueError("No interaction data found")
#         if self._Rpos_train is None:
#             raise ValueError("Train-test split not performed")
#         if self._Fu.nnz == 0:
#             raise ValueError("No user feature data found")
#         if self._Fi.nnz == 0:
#             raise ValueError("No item feature data found")

#         # Check coverage
#         users_with_features = set(self._Fu.row)
#         items_with_features = set(self._Fi.row)

#         if len(users_with_features) < self.n_users:
#             missing_users = self.n_users - len(users_with_features)
#             raise ValueError(f"{missing_users} users have no feature data")
#         if len(items_with_features) < self.n_items:
#             missing_items = self.n_items - len(items_with_features)
#             raise ValueError(f"{missing_items} items have no feature data")

#     @property
#     def Rpos(self) -> sp.coo_matrix:
#         """Get positive interactions matrix."""
#         return self._Rpos

#     @property
#     def Rpos_train(self) -> sp.coo_matrix:
#         """Get training positive interactions matrix."""
#         return self._Rpos_train

#     @property
#     def Rpos_test(self) -> sp.coo_matrix:
#         """Get testing positive interactions matrix."""
#         return self._Rpos_test

#     @property
#     def Rneg(self) -> sp.coo_matrix:
#         """Get negative interactions matrix."""
#         return self._Rneg

#     @property
#     def Fu(self) -> sp.coo_matrix:
#         """Get user features matrix."""
#         return self._Fu

#     @property
#     def Fi(self) -> sp.coo_matrix:
#         """Get item features matrix."""
#         return self._Fi


# """Base class for defining User-Item interaction data."""

# from typing import List, Optional, Tuple, Union

# import numpy as np
# import scipy.sparse as sp
# from tqdm import tqdm


# class UserItemData:
#     """Manages user-item interaction and feature data for recommendation systems."""

#     def __init__(
#         self,
#         name: str,
#         dtype: np.dtype = np.float32
#     ):
#         """Initialize UserItemData object with dynamic dimensions."""
#         self.name = name
#         self.dtype = dtype

#         # Initialize dimensions to zero - will be updated dynamically
#         self.n_users = 0
#         self.n_items = 0
#         self.n_user_features = 0
#         self.n_item_features = 0

#         # Initialize empty matrices
#         self._Rpos = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Rneg = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Rpos_train = None
#         self._Rpos_test = None
#         self._Fu = sp.coo_matrix((0, 0), dtype=dtype)
#         self._Fi = sp.coo_matrix((0, 0), dtype=dtype)

#     def __repr__(self) -> str:
#         """Return string representation of the dataset."""
#         n_total = self.Rpos_coo.nnz + self.Rneg_coo.nnz
#         pos_perc = (np.round(self.Rpos_coo.nnz * 100 / n_total, 1)
#                     if n_total > 0 else 0)
#         return (f"{self.__class__.__name__}(\n"
#                 f"  name='{self.name}'\n"
#                 f"  n_users={self.n_users:,}\n"
#                 f"  n_items={self.n_items:,}\n"
#                 f"  n_interactions={n_total:,} ({pos_perc}% pos)\n"
#                 f"  n_user_features={self.n_user_features:,}"
#                 f" (n_data={self._Fu.nnz:,})\n"
#                 f"  n_item_features={self.n_item_features:,}"
#                 f" (n_data={self._Fi.nnz:,})\n)")

#     def train_test_split(
#         self,
#         test_ratio: float = 0.2,
#         random_state: Optional[int] = None,
#         show_progress: bool = True
#     ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
#         """Efficiently split interactions with all users/items in training set."""
#         if not 0 < test_ratio < 1:
#             raise ValueError("test_ratio must be between 0 and 1")
#         if random_state is not None:
#             np.random.seed(random_state)

#         # Initialize matrices and process each user
#         csr = self.Rpos_coo.tocsr()
#         lil_train = sp.lil_matrix(csr.shape, dtype=self.dtype)
#         lil_test = sp.lil_matrix(csr.shape, dtype=self.dtype)
#         active_users = np.unique(csr.nonzero()[0])

#         # Process each user efficiently
#         for user in tqdm(active_users, desc="Splitting users",
#                          disable=not show_progress):
#             _, item_indices = csr[user].nonzero()
#             if len(item_indices) == 0:
#                 continue

#             # Split interactions
#             np.random.shuffle(item_indices)
#             n_test = min(max(int(len(item_indices) * test_ratio), 0),
#                          len(item_indices) - 1)
#             test_items, train_items = item_indices[:n_test], item_indices[n_test:]

#             # Get values and populate matrices
#             train_values = np.array([csr[user, i] for i in train_items])
#             test_values = np.array([csr[user, i] for i in test_items])
#             for i, val in zip(train_items, train_values):
#                 lil_train[user, i] = val
#             for i, val in zip(test_items, test_values):
#                 lil_test[user, i] = val

#         # Ensure all items are in the training set
#         train_items_present = set(lil_train.nonzero()[1])
#         all_items = set(self.Rpos_coo.col)
#         missing_items = all_items - train_items_present

#         if missing_items:
#             print(
#                 f"Found {len(missing_items)} items missing from training set")
#             test_coords = list(zip(*lil_test.nonzero()))
#             np.random.shuffle(test_coords)

#             for item in tqdm(missing_items, desc="Fixing missing items",
#                              disable=not show_progress):
#                 for user, test_item in test_coords:
#                     if test_item == item:
#                         # Move interaction to training
#                         lil_train[user, item] = lil_test[user, item]
#                         lil_test[user, item] = 0
#                         break

#         # Convert matrices and gather statistics
#         print("Converting matrices to COO format...")
#         self._Rpos_train = lil_train.tocoo()
#         self._Rpos_test = lil_test.tocoo()

#         # Calculate and print statistics
#         train_users, test_users = set(
#             self._Rpos_train.row), set(self._Rpos_test.row)
#         train_items, test_items = set(
#             self._Rpos_train.col), set(self._Rpos_test.col)
#         user_overlap = len(train_users.intersection(test_users))
#         item_overlap = len(train_items.intersection(test_items))

#         print(f"{self.__class__.__name__}: Train-test split completed\n"
#               f"  Train interactions: {self._Rpos_train.nnz:,}\n"
#               f"  Test interactions: {self._Rpos_test.nnz:,}\n"
#               f"  Users in train: {len(train_users):,}, in test: {len(test_users):,}\n"
#               f"  Items in train: {len(train_items):,}, in test: {len(test_items):,}\n"
#               f"  Users in both: {user_overlap:,} "
#               f"({user_overlap/len(train_users)*100:.1f}%)\n"
#               f"  Items in both: {item_overlap:,} "
#               f"({item_overlap/len(train_items)*100:.1f}%)")

#         return self._Rpos_train, self._Rpos_test

#     def _process_weights(
#         self,
#         weights: Optional[Union[float, List[float]]],
#         length: int
#     ) -> np.ndarray:
#         """Process and validate weights."""
#         if weights is None:
#             return np.ones(length, dtype=self.dtype)
#         if np.isscalar(weights):
#             return np.full(length, weights, dtype=self.dtype)
#         if len(weights) != length:
#             raise ValueError(
#                 f"Weight length ({len(weights)}) must match input length ({length})"
#             )
#         return np.array(weights, dtype=self.dtype)

#     def _update_matrix(
#         self,
#         old_matrix: sp.coo_matrix,
#         new_data: sp.coo_matrix,
#         new_shape: Tuple[int, int]
#     ) -> sp.coo_matrix:
#         """Helper method to update and resize sparse matrices."""
#         if old_matrix.nnz == 0:
#             return new_data

#         # Resize existing matrix if needed
#         if (old_matrix.shape[0] < new_shape[0] or old_matrix.shape[1] < new_shape[1]):
#             old_matrix = sp.coo_matrix(
#                 (old_matrix.data, (old_matrix.row, old_matrix.col)),
#                 shape=new_shape,
#                 dtype=self.dtype
#             )

#         # Combine matrices
#         result = old_matrix + new_data
#         result.eliminate_zeros()
#         return result

#     def add_positive_interactions(
#         self,
#         user_indices: List[int],
#         item_indices: List[int],
#         weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add positive interactions to the dataset and update dimensions."""
#         if len(user_indices) != len(item_indices):
#             raise ValueError("User and item indices must have equal length")

#         if len(user_indices) == 0:
#             return

#         # Update dimensions based on maximum indices
#         self.n_users = max(self.n_users, max(user_indices) + 1)
#         self.n_items = max(self.n_items, max(item_indices) + 1)

#         # Check for negative indices
#         if min(user_indices) < 0 or min(item_indices) < 0:
#             raise ValueError("User and item indices must be non-negative")

#         # Process weights and create matrix
#         values = self._process_weights(weights, len(user_indices))

#         interaction_matrix = sp.coo_matrix(
#             (values, (user_indices, item_indices)),
#             shape=(self.n_users, self.n_items),
#             dtype=self.dtype
#         )
#         interaction_matrix.eliminate_zeros()

#         # Update positive interaction matrix
#         self._Rpos = self._update_matrix(
#             self._Rpos,
#             interaction_matrix,
#             (self.n_users, self.n_items)
#         )

#         # Resize negative interaction matrix if necessary
#         if self._Rneg.nnz > 0 and (self._Rneg.shape[0] < self.n_users or
#                                    self._Rneg.shape[1] < self.n_items):
#             self._Rneg = sp.coo_matrix(
#                 (self._Rneg.data, (self._Rneg.row, self._Rneg.col)),
#                 shape=(self.n_users, self.n_items),
#                 dtype=self.dtype
#             )

#     def add_negative_interactions(
#         self,
#         user_indices: List[int],
#         item_indices: List[int],
#         weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """
#         Add negative interactions to the dataset.
#         Only users with existing positive interactions can have negative interactions.
#         """
#         if len(user_indices) != len(item_indices):
#             raise ValueError("User and item indices must have equal length")

#         if len(user_indices) == 0:
#             return

#         # Get users with positive interactions
#         users_with_pos = set(self._Rpos.row)

#         # Filter out users without positive interactions
#         valid_indices = [(i, u, item) for i, (u, item) in
#                          enumerate(zip(user_indices, item_indices))
#                          if u in users_with_pos]

#         if len(valid_indices) < len(user_indices):
#             n_ignored = len(user_indices) - len(valid_indices)
#             print(f"Ignoring {n_ignored} negative interactions from users with no "
#                   f"positive interactions ({n_ignored/len(user_indices):.1%} of total)")

#             if not valid_indices:
#                 print("No valid negative interactions remaining, skipping")
#                 return

#             # Extract valid indices
#             valid_idx, user_indices, item_indices = zip(*valid_indices)
#             if weights is not None and not np.isscalar(weights):
#                 weights = [weights[i] for i in valid_idx]

#         # Check bounds and negative indices
#         if max(user_indices) >= self.n_users or max(item_indices) >= self.n_items:
#             raise ValueError("User/item indices exceed current dimensions")

#         if min(user_indices) < 0 or min(item_indices) < 0:
#             raise ValueError("User and item indices must be non-negative")

#         # Process weights and create matrix
#         values = self._process_weights(weights, len(user_indices))

#         interaction_matrix = sp.coo_matrix(
#             (values, (user_indices, item_indices)),
#             shape=(self.n_users, self.n_items),
#             dtype=self.dtype
#         )
#         interaction_matrix.eliminate_zeros()

#         # Update negative interaction matrix
#         self._Rneg = self._update_matrix(
#             self._Rneg,
#             interaction_matrix,
#             (self.n_users, self.n_items)
#         )

#     def add_user_features(
#         self,
#         user_indices: List[int],
#         feature_indices: List[int],
#         feature_weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add user features and update dimensions."""
#         if len(user_indices) != len(feature_indices):
#             raise ValueError("User and feature indices must have equal length")

#         if len(user_indices) == 0:
#             return

#         # Update dimensions and validate
#         self.n_users = max(self.n_users, max(user_indices) + 1)
#         self.n_user_features = max(
#             self.n_user_features, max(feature_indices) + 1)

#         if min(user_indices) < 0 or min(feature_indices) < 0:
#             raise ValueError("User and feature indices must be non-negative")

#         # Process weights and create matrix
#         values = self._process_weights(feature_weights, len(user_indices))

#         feature_matrix = sp.coo_matrix(
#             (values, (user_indices, feature_indices)),
#             shape=(self.n_users, self.n_user_features),
#             dtype=self.dtype
#         )
#         feature_matrix.eliminate_zeros()

#         # Update user feature matrix
#         self._Fu = self._update_matrix(
#             self._Fu,
#             feature_matrix,
#             (self.n_users, self.n_user_features)
#         )

#     def add_item_features(
#         self,
#         item_indices: List[int],
#         feature_indices: List[int],
#         feature_weights: Optional[Union[float, List[float]]] = None
#     ) -> None:
#         """Add item features and update dimensions."""
#         if len(item_indices) != len(feature_indices):
#             raise ValueError("Item and feature indices must have equal length")

#         if len(item_indices) == 0:
#             return

#         # Update dimensions and validate
#         self.n_items = max(self.n_items, max(item_indices) + 1)
#         self.n_item_features = max(
#             self.n_item_features, max(feature_indices) + 1)

#         if min(item_indices) < 0 or min(feature_indices) < 0:
#             raise ValueError("Item and feature indices must be non-negative")

#         # Process weights and create matrix
#         values = self._process_weights(feature_weights, len(item_indices))

#         feature_matrix = sp.coo_matrix(
#             (values, (item_indices, feature_indices)),
#             shape=(self.n_items, self.n_item_features),
#             dtype=self.dtype
#         )
#         feature_matrix.eliminate_zeros()

#         # Update item feature matrix
#         self._Fi = self._update_matrix(
#             self._Fi,
#             feature_matrix,
#             (self.n_items, self.n_item_features)
#         )

#     def validate_dataset(self) -> None:
#         """Validate the entire dataset."""
#         # Check for required data
#         if self.Rpos_coo.nnz == 0:
#             raise ValueError("No interaction data found")
#         if self._Rpos_train is None:
#             raise ValueError("Train-test split not performed")
#         if self._Fu.nnz == 0:
#             raise ValueError("No user feature data found")
#         if self._Fi.nnz == 0:
#             raise ValueError("No item feature data found")

#         # Check coverage
#         users_with_features = set(self._Fu.row)
#         items_with_features = set(self._Fi.row)

#         if len(users_with_features) < self.n_users:
#             missing_users = self.n_users - len(users_with_features)
#             raise ValueError(f"{missing_users} users have no feature data")
#         if len(items_with_features) < self.n_items:
#             missing_items = self.n_items - len(items_with_features)
#             raise ValueError(f"{missing_items} items have no feature data")

#     @property
#     def Rpos_coo(self) -> sp.coo_matrix:
#         """Get positive interactions matrix."""
#         return self._Rpos

#     @property
#     def Rpos_train_coo(self) -> sp.coo_matrix:
#         """Get training positive interactions matrix."""
#         return self._Rpos_train

#     @property
#     def Rpos_test_coo(self) -> sp.coo_matrix:
#         """Get testing positive interactions matrix."""
#         return self._Rpos_test

#     @property
#     def Rneg_coo(self) -> sp.coo_matrix:
#         """Get negative interactions matrix."""
#         return self._Rneg

#     @property
#     def Fu_coo(self) -> sp.coo_matrix:
#         """Get user features matrix."""
#         return self._Fu

#     @property
#     def Fi_coo(self) -> sp.coo_matrix:
#         """Get item features matrix."""
#         return self._Fi

# class UserItemData:
#     """
#     Base class for setting up data for recomednation system
#     """

#     def __init__(
#         self,
#         name: str,
#         n_users: int,
#         n_items: int,
#         n_user_features: int,
#         n_item_features: int
#     ):
#         """Initiate"""
#         # basics
#         # self.printit('Initiating dataset..')
#         self.name = name
#         self.sp_dtype = np.float32
#         self.n_users = int(n_users)
#         self.n_items = int(n_items)
#         self.n_user_features = int(n_user_features)
#         self.n_item_features = int(n_item_features)

#         # interaction data containing matrices
#         self._Rpos = coo_array(
#             (self.n_users, self.n_items),
#             dtype=self.sp_dtype
#         )
#         self._Rneg = coo_array(
#             (self.n_users, self.n_items),
#             dtype=self.sp_dtype
#         )

#         # train test split
#         self._Rpos_train = None
#         self._Rpos_test = None

#         # feature contianing matrices
#         self._Fu = coo_array(
#             (self.n_users, self.n_user_features),
#             dtype=self.sp_dtype
#         )
#         self._Fi = coo_array(
#             (self.n_items, self.n_item_features),
#             dtype=self.sp_dtype
#         )

#     def __str__(self):
#         n_total = self.Rpos_coo.nnz + self.Rneg_coo.nnz
#         pos_perc = np.round(self.Rpos_coo.nnz*100/n_total, 1)
#         return (f'{self.__class__.__name__}(\n'
#                 f'  n_users={self.n_users:,}\n'
#                 f'  n_items={self.n_items:,}\n'
#                 f'  n_interactions={n_total:,} ({pos_perc}% pos)\n'
#                 f'  n_user_features={self.n_user_features:,}'
#                 f' (n_data={self._Fu.nnz:,})\n'
#                 f'  n_item_features={self.n_item_features:,}'
#                 f' (n_data={self._Fi.nnz:,})\n)'
#                 )

#     def train_test_split(self, test_ratio: float = 0.2):
#         """Train test split"""
#         self.printit(
#             f'Performing train({1-test_ratio})-test({test_ratio}) split..')
#         n_data = self.Rpos_coo.nnz
#         n_test = int(test_ratio*n_data)
#         all_ids = list(range(n_data))
#         test_ids = list(np.random.randint(low=0, high=n_data-1, size=n_test))
#         train_ids = list(set(all_ids) - set(test_ids))

#         # training data
#         mask = train_ids
#         self._Rpos_train = coo_array(
#             (self.Rpos_coo.data[mask],
#              (self.Rpos_coo.row[mask], self.Rpos_coo.col[mask])),
#             shape=(self.n_users, self.n_items),
#             dtype=self.sp_dtype
#         )
#         n_users = len(np.unique(self._Rpos_train.row))
#         n_items = len(np.unique(self._Rpos_train.col))
#         self.printit(f'Found {n_users:,}/{self.n_users:,} users in Rpos-train')
#         self.printit(f'Found {n_items:,}/{self.n_items:,} items in Rpos-train')

#         # test data
#         mask = test_ids
#         self._Rpos_test = coo_array(
#             (self.Rpos_coo.data[mask],
#              (self.Rpos_coo.row[mask], self.Rpos_coo.col[mask])),
#             shape=(self.n_users, self.n_items),
#             dtype=self.sp_dtype
#         )

#         n_users = len(np.unique(self._Rpos_test.row))
#         n_items = len(np.unique(self._Rpos_test.col))
#         self.printit(f'Found {n_users:,}/{self.n_users:,} users in Rpos-test')
#         self.printit(f'Found {n_items:,}/{self.n_items:,} items in Rpos-test')

#     def add_interactions(
#             self,
#             user_indices: list[int],
#             item_indices: list[int],
#             weights: list[float] | None = None,
#             positive: bool = True
#     ):
#         """assign data to the dataset"""
#         if len(user_indices) != len(item_indices):
#             self.value_err('user/item input len mismatch!')

#         self.check_user_indices(user_indices)
#         self.check_item_indices(item_indices)

#         # update the sparse interaction matrix
#         values = self._get_weights(weights, len(user_indices))
#         Rmat = coo_array(
#             (values, (user_indices, item_indices)),
#             shape=(self.n_users, self.n_items),
#             dtype=self.sp_dtype
#         )
#         Rmat.eliminate_zeros()
#         if positive:
#             self._Rpos = Rmat
#         else:
#             self._Rneg = Rmat

#     def add_user_features(
#             self,
#             user_indices: list[int],
#             feature_indices: list[int],
#             feature_weights: list[int] | None = None

#     ):
#         """add user feature data"""

#         # check input compaitbility
#         if len(user_indices) != len(feature_indices):
#             raise ValueError('user/feature input len mismatch!')

#         self.check_user_indices(user_indices)
#         if np.amax(feature_indices) >= self.n_user_features:
#             raise ValueError(
#                 f'Max user feature index ({np.amax(feature_indices)})'
#                 f' >= n_user_features ({self.n_user_features})'
#             )

#         # fill the sparse matrix with data
#         values = self._get_weights(feature_weights, len(user_indices))
#         self._Fu = coo_array(
#             (values, (user_indices, feature_indices)),
#             shape=(self.n_users, self.n_user_features),
#             dtype=self.sp_dtype
#         )
#         self._Fu.eliminate_zeros()

#     def add_item_features(
#             self,
#             item_indices: list[int],
#             feature_indices: list[int],
#             feature_weights: list[int] | None = None

#     ):
#         """add item feature data"""

#         # check input compaitbility
#         if len(item_indices) != len(feature_indices):
#             raise ValueError('user/feature input len mismatch!')

#         self.check_item_indices(item_indices)
#         if np.amax(feature_indices) >= self.n_item_features:
#             raise ValueError(
#                 f'Max item feature index ({np.amax(feature_indices)})'
#                 f' >= n_item_features ({self.n_item_features})'
#             )

#         # fill the sparse matrix with data
#         values = self._get_weights(feature_weights, len(item_indices))
#         self._Fi = coo_array(
#             (values, (item_indices, feature_indices)),
#             shape=(self.n_items, self.n_item_features),
#             dtype=self.sp_dtype
#         )
#         self._Fi.eliminate_zeros()

#     def check_if_initiated_properly(self):
#         """Check compatibility of inputs"""
#         if self.Rpos_coo.nnz == 0:
#             self.value_err('No interaction data found!')

#         if self.Rpos_train_coo.nnz is None:
#             self.value_err('Train-test split not performed!')

#         if self._Fu.nnz == 0:
#             self.value_err('No user feature data found!')

#         if self._Fi.nnz == 0:
#             self.value_err('No item feature data found!')

#         # check all users have slteast one feature
#         n_users_without_feats = self.n_users - len(set(self._Fu.row))
#         if n_users_without_feats > 0:
#             self.value_err(
#                 'Need atleast one feature for each user entry. '
#                 f'{n_users_without_feats} users have no feature data.'
#             )

#         # check all items have atleast one feature
#         n_items_without_feats = self.n_items - len(set(self._Fi.row))
#         if n_items_without_feats > 0:
#             self.value_err(
#                 'Need atleast one feature for each item entry. '
#                 f'{n_items_without_feats} item shave no feature data.'
#             )

#     def check_user_indices(self, u_ids: list[int]) -> None:
#         """check compatibility of user indices"""
#         if np.amin(u_ids) < 0:
#             self.value_err('user indices cannot be negative!')

#         if np.amax(u_ids) >= self.n_users:
#             self.value_err(
#                 f'max user index ({np.amax(u_ids)})'
#                 f' >= n_users ({self.n_users})'
#             )

#     def check_item_indices(self, i_ids: list[int]) -> None:
#         """check compatibility of user indices"""
#         if np.amin(i_ids) < 0:
#             self.value_err('item indices cannot be negative!')

#         if np.amax(i_ids) >= self.n_items:
#             self.value_err(
#                 f'max item index ({np.amax(i_ids)})'
#                 f' >= n_items ({self.n_item})'
#             )

#     def _get_weights(self, weights, length: int) -> ndarray:
#         """Handle weights"""
#         if weights is not None:
#             if np.isscalar(weights):
#                 # assert -1. <= weights <= 1., 'Weights should be in [-1,1]'
#                 values = np.repeat(weights, length)
#             else:
#                 if length != len(weights):
#                     self.value_err(
#                         f'len(weights) ({len(weights)})'
#                         f' !=  {length}'
#                     )
#                 # assert np.amin(weights) >= -1., 'Weights should be >= -1.'
#                 # assert np.amax(weights) <= 1., 'Weights should be <= 1.'
#                 values = weights
#         else:
#             # self.printit('Setting interaction weights to 1.')
#             values = np.repeat(1., length)
#         return values

#     def value_err(self, istr: str) -> None:
#         """Raise value Error"""
#         raise ValueError(f'{self.__class__.__name__}: {istr}')

#     def printit(self, istr: str) -> None:
#         """Raise value Error"""
#         print(f'{self.__class__.__name__}: {istr}', flush=True)

#     @property
#     def Rpos_coo(self) -> sparray:
#         """Returns interaction matrix"""
#         return self._Rpos

#     @property
#     def Rpos_train_coo(self) -> sparray:
#         """Returns interaction matrix"""
#         return self._Rpos_train

#     @property
#     def Rpos_test_coo(self) -> sparray:
#         """Returns interaction matrix"""
#         return self._Rpos_test

#     @property
#     def Rneg_coo(self) -> sparray:
#         """Returns interaction matrix"""
#         return self._Rneg

#     @property
#     def Fu_coo(self) -> sparray:
#         """Returns user feature matrix"""
#         return self._Fu

#     @property
#     def Fi_coo(self) -> sparray:
#         """Returns item feature matrix"""
#         return self._Fi

    # @property
    # def Rpos_coo(self) -> sparray:
    #     """Return sparse COO matrix with positive interactions"""
    #     mask = self._R.data > 0.
    #     return coo_array(
    #         (self._R.data[mask], (self._R.row[mask], self._R.col[mask])),
    #         shape=(self.n_users, self.n_items),
    #         dtype=self.sp_dtype
    #     )

    # @property
    # def Rneg_coo(self) -> spmatrix:
    #     """Return sparse COO matrix with negative interactions"""
    #     mask = self._R.data < 0.
    #     return coo_array(
    #         (self._R.data[mask], (self._R.row[mask], self._R.col[mask])),
    #         shape=(self.n_users, self.n_items),
    #         dtype=self.sp_dtype
    #     )

    # @property
    # def mat_
    #     def create_test_matrix(
    #         self,
    #         items_index: list[int],
    #         users_index: list[int]
    #     ):
    #         """Create test matrix"""
    #         self._mat_test = csr_matrix(
    #             # (np.ones((len(items_index),)), (users_index, items_index)),
    #             (np.repeat(True, len(items_index)), (users_index, items_index)),
    #             shape=self._Rmat.shape,
    #             dtype=self.dtype
    #         )

    #     def generate_train_test_split(
    #         self,
    #         user_test_ratio: float = 0.2,
    #         seed: int = 1234
    #     ):
    #         """Split the UI matrix into train and test"""
    #         print('Generating train-test split..', end="")
    #         assert user_test_ratio <= 0.5, 'user_test_ratio should be in [0,0.5]'
    #         assert user_test_ratio >= 0.0, 'user_test_ratio should be in [0,0.5]'

    #         if user_test_ratio < 1e-3:
    #             print('Warning: Test matrix is set as empty/all-zeros', flush=True)
    #             self._mat_test = csr_matrix(self._Rmat.shape, dtype=self.dtype)
    #             _mat_train = self._Rmat.copy()
    #         else:
    #             # min number of item interactions required per user
    #             min_interactions = int(1/user_test_ratio) + 1

    #             # find those users that haave atleast min_interactions
    #             num_interactions = np.asarray(self._Rmat.sum(axis=1)).reshape(-1)
    #             valid_users = np.where(num_interactions > min_interactions)[0]

    #             # initiate test and train matrices
    #             np.random.seed(seed=seed)
    #             self._mat_test = dok_matrix(self._Rmat.shape, dtype=self.dtype)
    #             _mat_train = self._Rmat.copy().todok()

    #             # iterate over each user and split its interactions into test/train
    #             for ith_user in valid_users:
    #                 items_ith_user = self._mat[ith_user].indices
    #                 test_size = int(np.ceil(user_test_ratio * items_ith_user.size))
    #                 ith_items = np.random.choice(
    #                     items_ith_user,
    #                     size=test_size,
    #                     replace=False
    #                 )
    #                 self._mat_test[ith_user, ith_items] = True
    #                 _mat_train[ith_user, ith_items] = False
    #             self._mat_test = self._Rmat_test.tocsr()
    #             self._mat = _mat_train.tocsr()
    #             print('done', flush=True)

    #         # check if train+test=original UI mat
    #         if (self._Rmat_train + self._Rmat_test != self._Rmat).nnz != 0:
    #             raise RuntimeError('Issue with test/train split')

    #     def __str__(self) -> SyntaxWarning:
    #         """Print output"""
    #         out_str = f'\n---{self.name}---\n'

    #         # training mat
    #         ival = np.unique(self._Rmat_train.tocoo().row).size
    #         out_str += f'# of users, train: {ival}/{self.num_users}\n'
    #         ival = np.unique(self._Rmat_train.tocoo().col).size
    #         out_str += f'# of items, train: {ival}/{self.num_items}\n'

    #         # test mat
    #         if self._Rmat_test is not None:
    #             ival = np.unique(self._Rmat_test.tocoo().row).size
    #             out_str += f'# of users, test: {ival}/{self.num_users}\n'
    #             ival = np.unique(self._Rmat_test.tocoo().col).size
    #             out_str += f'# of items, test: {ival}/{self.num_items}\n'
    #         # out_str += f'# of interactions: {self._Rmat.nnz}\n'

    #         # interations
    #         out_str += f'# of interactions, train: {self._Rmat_train.nnz}\n'
    #         if self._Rmat_test is not None:
    #             out_str += f'# of interactions, test: {self._Rmat_test.nnz}\n'

    #         # sparsity
    #         # train_s = self._Rmat_train.nnz / (self.num_users * self.num_items)
    #         # test_s = self._Rmat_test.nnz / (self.num_users * self.num_items)
    #         # out_str += f'Sparsity in the train/test mat: {train_s}/{test_s}\n'

    #         # memory usage
    #         train_mb = np.around(self._Rmat_train.data.nbytes / 1024 / 1024, 2)
    #         test_mb = np.around(self._Rmat_test.data.nbytes / 1024 / 1024, 2)
    #         out_str += f'Memory used by train/test mat: {train_mb}/{test_mb} MB'
    #         return out_str

    #     @property
    #     def mat(self) -> spmatrix:
    #         """Number of users"""
    #         return self._mat

    #     @property
    #     def mat_test(self) -> spmatrix:
    #         """Number of users"""
    #         return self._mat_test

    #     @property
    #     def mat_train(self) -> spmatrix:
    #         """Number of users"""
    #         return self._mat

    #     @property
    #     def num_users(self) -> int:
    #         """Number of users"""
    #         return self._Rmat.shape[0]

    #     @property
    #     def num_items(self) -> int:
    #         """Number of users"""
    #         return self._Rmat.shape[1]

    #     @property
    #     def active_users(self) -> ndarray:
    #         """Index of users with atleast one interaction"""
    #         return np.where(np.asarray(self._Rmat.sum(axis=1)).reshape(-1) > 0)[0]

    #     @property
    #     def active_items(self) -> ndarray:
    #         """Index of items with atleast one interaction"""
    #         return np.where(np.asarray(self._Rmat.sum(axis=0)).reshape(-1) > 0)[0]

    #     @property
    #     def num_users_active(self) -> int:
    #         """Number of users"""
    #         return self.active_users.size

    #     @property
    #     def num_items_active(self) -> int:
    #         """Number of users"""
    #         return self.active_items.size

    #     @property
    #     def sparsity(self) -> float:
    #         """Get the sparsity"""
    #         outs = self._Rmat.nnz * 1 / (self.num_users * self.num_items)
    #         return np.around(outs, 6)

    # # Junk
    #     # def users_sorted_by_activity(self, count: int | None = None):
    #     #     """return the most -count- active users"""
    #     #     # get number of interactions for each user
    #     #     count_vector = np.asarray(self._mat.sum(axis=1)).reshape(-1)
    #     #     # get the index of users with most interactions at 0
    #     #     sorted_user_list = count_vector.argsort()[::-1]
    #     #     if count is not None:
    #     #         sorted_user_list = sorted_user_list[:count]
    #     #     return sorted_user_list

    #     # def get_ndcg_metric_user(
    #     #     self,
    #     #     user_idx: int,
    #     #     user_mat,
    #     #     item_mat,
    #     #     num_items: int,
    #     #     test: bool = True,
    #     #     truncate: bool = True
    #     # ):
    #     #     """Computes NDCG score for this user"""
    #     #     data_mat = self._mat_test if test else self._mat_train
    #     #     test_inds = list(data_mat[user_idx].indices)
    #     #     num_items = min(len(test_inds), num_items) if truncate else num_items
    #     #     exclude_liked = True if test else False
    #     #     top_items = self.get_top_items_for_this_user(
    #     #         user_idx=user_idx,
    #     #         user_mat=user_mat,
    #     #         item_mat=item_mat,
    #     #         num_items=num_items,
    #     #         exclude_liked=exclude_liked
    #     #     )
    #     #     ndcg_score = compute_ndcg(
    #     #         ranked_item_idx=np.where(np.isin(top_items, test_inds))[0],
    #     #         K=num_items,
    #     #         wgt_fun=np.log2
    #     #     )
    #     #     return ndcg_score

    #     # # def get_ndcg_metric(
    #     #     self,
    #     #     user_mat,
    #     #     item_mat,
    #     #     num_items: int,
    #     #     num_users: int = 100,
    #     #     test: bool = True,
    #     #     truncate: bool = False,
    #     #     ncores:int=1
    #     # ):
    #     #     """Averaged NDCG across all users"""
    #     #     pfunc = partial(
    #     #         self.get_ndcg_metric_user,
    #     #         user_mat=user_mat,
    #     #         item_mat=item_mat,
    #     #         num_items=num_items,
    #     #         test=test,
    #     #         truncate=truncate
    #     #     )
    #     #     ndcg_scores = []
    #     #     chosen_users = np.random.choice(
    #     #         self.num_users,
    #     #         size=num_users
    #     #     )
    #     #     if self.ncores <= 1:
    #     #         for idx in chosen_users:
    #     #             ndcg_scores.append(pfunc(idx))
    #     #     else:
    #     #         with mp.Pool(processes=self.ncores) as p:
    #     #             ndcg_scores = p.map(pfunc, chosen_users)
    #     #     return np.mean(ndcg_scores)

    #     # def get_similar(
    #     #     self,
    #     #     feature_mat,
    #     #     idx: int,
    #     #     count: int = 5
    #     # ):
    #     #     """
    #     #     Get similar pairs, items or users
    #     #     """
    #     #     # cosine distance is proportional to normalized euclidean distance,
    #     #     # thus we normalize the item vectors and use euclidean metric so
    #     #     # we can use the more efficient kd-tree for nearest neighbor search;
    #     #     # also the item will always to nearest to itself, so we add 1 to
    #     #     # get an additional nearest item and remove itself at the end
    #     #     normed_factors = normalize(feature_mat)
    #     #     knn = NearestNeighbors(n_neighbors=count + 1, metric='euclidean')
    #     #     knn.fit(normed_factors)
    #     #     normed_factors = np.atleast_2d(normed_factors[idx])
    #     #     _, inds = knn.kneighbors(normed_factors)
    #     #     similar_inds = list(np.squeeze(inds.astype(np.uint32)))
    #     #     similar_inds = [ix for ix in similar_inds if ix != idx]
    #     #     return similar_inds

    # # cf_8user_10item = UserItemInteractions(
    # #     users=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
    # #            5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
    # #     items=[1, 2, 3, 4, 5, 1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4,
    # #            6, 7, 8, 9, 10, 6, 7, 8, 9, 7, 8, 9, 10, 7, 8, 9]
    # # )

    # # cf_4user_5item = UserItemInteractions(
    # #     users=[1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
    # #     items=[1, 2, 3, 5, 1, 2, 4, 2, 3, 3],
    # #     min_num_rating_per_user=0,
    # #     min_num_rating_per_item=0,
    # # )

    # # # def get_idx(entry):
    # # #     print(entry[self.item_col])
    # # #     item_idx = pd.Index(
    # # #         self.df_item[self.item_col]).get_loc(entry[self.item_col].values)
    # # #     user_idx = pd.Index(
    # # #         self.df_user[user_col]).get_loc(entry[user_col].values)
    # # #     return item_idx, user_idx
    # # # indices_tuple = self.df.swifter.apply(get_idx, axis=1)
    # # # item_inds, user_inds = zip(*indices_tuple)
    # # # user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
    # # # self.df_user = self.df_user[user_bool]
    # # # self.df_user.sort_values(by=num_weights_col, ascending=False,
    # # #                          inplace=True)
    # # # self.df_user.reset_index(drop=True, inplace=True)
    # # # item_bool = self.df_item[self.item_col].isin(self.df[self.item_col].unique())
    # # # self.df_item = self.df_item[item_bool]
    # # # self.df_item.sort_values(by=num_weights_col, ascending=False,
    # #                          inplace=True)
    # # self.df_item.reset_index(drop=True, inplace=True)

    # # number of user and items

    # # # construct the user - item matrix
    # # self.df_user.reset_index(drop=False, inplace=True)
    # # self.df_user.set_index(user_col, drop=True, inplace=True)
    # # self.df['UserIdx'] = self.df_user.loc[self.df[user_col]]['index'].values
    # # self.df_user.drop(columns=('index'))

    # # self.df_item.reset_index(drop=False, inplace=True)
    # # self.df_item.set_index(self.item_col, drop=True, inplace=True)
    # # self.df['ItemIdx'] = self.df_item.loc[self.df[self.item_col]]['index'].values
    # # self.df_item.drop(columns=('index'))
    # # u_inds = self.df_user[user_col].cat.codes.values
    # # i_inds = self.df_item[self.item_col].cat.codes.values
    # # self.df_user['Numweights_Train'] = self._mat_train.sum(axis=1)[u_inds, 0]
    # # self.df_user['Numweights_Test'] = self._mat_test.sum(axis=1)[u_inds, 0]
    # # self.df_item['Numweights_Train'] = self._mat_train.sum(axis=0)[0, i_inds]
    # # self.df_item['Numweights_Test'] = self._mat_test.sum(axis=0)[0, i_inds]

    # # def get_top_items_for_this_user(
    # #     self,
    # #     R_est,
    # #     user: int,
    # #     exclude_training: bool = True
    # # ):
    # #     """Returns top products for this user"""
    # #     user_pred =
    # #     items_for_this_user = R_est[user, :]
    #   # # trim the interaction data based on min user item conditions
    #         # ubool = (self.df_user[num_weights_col] >= min_num_rating_per_user)
    #         # selected_users = self.df_user.loc[ubool, user_col]
    #         # ibool = (self.df_item[num_weights_col] >= min_num_rating_per_item)
    #         # selected_items = self.df_item.loc[ibool, self.item_col]
    #         # user_bool = (self.df[user_col].isin(selected_users))
    #         # item_bool = (self.df[self.item_col].isin(selected_items))
    #         # self.df = self.df[item_bool & user_bool].reset_index(drop=True)
    #         # user_bool = self.df_user[user_col].isin(self.df[user_col].unique())
    #         # self.df_user = self.df_user[user_bool]
    #         # item_bool = self.df_item[self.item_col].isin(self.df[self.item_col].unique())
    #         # self.df_item = self.df_item[item_bool]

    #         # # slice the user and item dataframes
    #         # iterator = zip([self.df_item, self.df_user], [self.item_col, user_col],
    #         #                [self.item_col, user_col])
    #         # for idf, id_colname, idx_colname in iterator:
    #         #     # idf = idf[idf[id_colname].isin(
    #         #     #     self.df[id_colname].unique())].copy()
    #         #     idf.sort_values(by=num_weights_col, ascending=False, inplace=True)
    #         #     idf.reset_index(drop=True, inplace=True)
    #         #     idf.reset_index(drop=False, inplace=True)
    #         #     idf.set_index(id_colname, drop=True, inplace=True)
    #         #     self.df[idx_colname] = idf.loc[self.df[id_colname]]['index'].values
    #         #     idf.rename(columns={'index': idx_colname}, inplace=True)
    #         #     idf.reset_index(drop=False, inplace=True)
    #         #     idf.set_index(idx_colname, drop=True, inplace=True)

    #         #     self.user_col = 'user_idx'
    #         # self.item_col = 'item_idx'
    #         # num_weights_col = 'num_interactions'
    #         # self.df = pd.DataFrame({
    #         #     self.user_col: users_index,
    #         #     self.item_col: items_index
    #         # }, dtype=int)

    #         # if timestamps is not None:
    #         #     self.df['Timestamp'] = timestamps

    #         # # create user and item dframes
    #         # agg_flag = {num_weights_col: (self.item_col, 'count')}
    #         # self.df_user = self.df.groupby(self.user_col).agg(**agg_flag)
    #         # self.df_user.sort_values(by=num_weights_col, inplace=True)
    #         # agg_flag = {num_weights_col: (self.user_col, 'count')}
    #         # self.df_item = self.df.groupby(self.item_col).agg(**agg_flag)
    #         # self.df_item.sort_values(by=num_weights_col, inplace=True)
