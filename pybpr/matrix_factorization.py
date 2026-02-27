#!/usr/bin/env python
"""
Base class for defining matrix factorization model for recommendation systems.

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

from typing import Tuple, Optional, Union

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class HybridMF(nn.Module):
    """Hybrid Matrix Factorization model with user and item features."""

    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_latent: int,
        sparse: bool = False,
        use_global_bias: bool = True,
        use_user_bias: bool = False,
        dropout: float = 0.0,
        init_std: Optional[float] = None,
        activation: Optional[str] = None,
    ) -> None:
        """Initialize hybrid matrix factorization model.

        Args:
            n_user_features: Number of user features/vocab size
            n_item_features: Number of item features/vocab size
            n_latent: Dimensionality of latent factors
            sparse: Whether to use sparse embeddings
            use_global_bias: Whether to include global bias term
            use_user_bias: Whether to include user bias terms
            dropout: Dropout rate for regularization (0.0 = no dropout)
            init_std: Standard deviation for weight initialization
            activation: Activation function ('relu', 'tanh', 'sigmoid', or None)
        """
        super().__init__()
        self.n_latent = n_latent
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.dropout = dropout
        self.init_std = init_std or 0.1
        self.use_user_bias = use_user_bias
        self.use_global_bias = use_global_bias

        # Define embeddings for users and items
        self.user_latent = nn.Embedding(
            num_embeddings=n_user_features,
            embedding_dim=n_latent,
            sparse=sparse,
        )
        self.item_latent = nn.Embedding(
            num_embeddings=n_item_features,
            embedding_dim=n_latent,
            sparse=sparse,
        )
        self.item_biases = nn.Embedding(
            num_embeddings=n_item_features,
            embedding_dim=1,
            sparse=sparse
        )
        if use_user_bias:
            self.user_biases = nn.Embedding(
                num_embeddings=n_user_features,
                embedding_dim=1,
                sparse=sparse
            )

        if use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1))

        # Dropout layer for regularization
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)

        # Activation function
        self.activation = self._get_activation(activation)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(
            self,
            activation: Optional[str]
    ) -> Optional[nn.Module]:
        """Get activation function by name."""
        if activation is None:
            return None

        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        return activation_map[activation.lower()]

    def _initialize_weights(self) -> None:
        """Initialize model weights and biases with appropriate values."""
        # Initialize latent factors with Xavier-like initialization
        nn.init.normal_(self.user_latent.weight.data, 0, self.init_std)
        nn.init.normal_(self.item_latent.weight.data, 0, self.init_std)
        nn.init.zeros_(self.item_biases.weight.data)

        if self.use_user_bias:
            nn.init.zeros_(self.user_biases.weight.data)

    def _scipy_to_torch_sparse(
        self,
        scipy_matrix: sp.spmatrix,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        # Convert to COO format for PyTorch compatibility
        coo = scipy_matrix.tocoo()

        # Create indices and values tensors
        indices = torch.from_numpy(
            np.vstack((coo.row, coo.col)).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))

        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32
        )

        if device is not None:
            sparse_tensor = sparse_tensor.to(device)

        return sparse_tensor.coalesce()

    def _safe_sparse_mm(
        self,
        features: Union[torch.Tensor, sp.spmatrix],
        dense_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Safely perform sparse matrix multiplication with error handling"""
        # Handle scipy sparse matrices
        if sp.issparse(features):
            features = self._scipy_to_torch_sparse(
                features, device=dense_tensor.device
            )

        # Handle PyTorch tensors
        if isinstance(features, torch.Tensor):
            # Ensure tensors are on the same device
            if features.device != dense_tensor.device:
                features = features.to(dense_tensor.device)

            # Perform multiplication based on sparsity
            if features.is_sparse:
                return torch.sparse.mm(features, dense_tensor)
            else:
                return torch.mm(features, dense_tensor)

        raise TypeError(f"Unsupported features type: {type(features)}")

    def forward(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Predict rating/preference score for user-item pairs.

        Args:
            user_features: Sparse/dense tensor or scipy sparse matrix of shape
                (batch_size, n_user_features)
            item_features: Sparse/dense tensor or scipy sparse matrix of shape
                (batch_size, n_item_features)

        Returns:
            Predictions tensor of shape (batch_size,)
        """
        # Extract latent vectors and compute interactions
        user_latent = self._safe_sparse_mm(
            user_features, self.user_latent.weight)
        item_latent = self._safe_sparse_mm(
            item_features, self.item_latent.weight)

        # Apply dropout for regularization
        if self.dropout > 0.0 and self.training:
            user_latent = self.dropout_layer(user_latent)
            item_latent = self.dropout_layer(item_latent)

        # Apply activation function to latent representations
        if self.activation is not None:
            user_latent = self.activation(user_latent)
            item_latent = self.activation(item_latent)

        # Compute latent interactions (element-wise multiplication and sum)
        latent_interaction = torch.sum(user_latent * item_latent, dim=-1)

        # Add bias terms starting with latent interactions
        predictions = latent_interaction

        # Add item bias
        item_bias = self._safe_sparse_mm(
            item_features, self.item_biases.weight)
        predictions = predictions + item_bias.squeeze(-1)

        # Add user bias (optional)
        if self.use_user_bias:
            user_bias = self._safe_sparse_mm(
                user_features, self.user_biases.weight
            )
            predictions = predictions + user_bias.squeeze(-1)

        # Add global bias (optional)
        if self.use_global_bias:
            predictions = predictions + self.global_bias

        return predictions

    def predict(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
        item_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict ratings with optional item batching.

        Args:
            user_features: Sparse/dense or scipy sparse matrix
                (n_users, n_user_features)
            item_features: Sparse/dense or scipy sparse matrix
                (n_items, n_item_features)
            item_batch_size: Batch size for items (None = no batching)

        Returns:
            Prediction matrix of shape (n_users, n_items)
        """
        with torch.no_grad():
            # Get number of items
            n_items = (
                item_features.shape[0] if sp.issparse(item_features)
                else item_features.size(0)
            )

            # Batch items for large catalogs
            if item_batch_size is not None and n_items > item_batch_size:
                all_preds = []
                for i in range(0, n_items, item_batch_size):
                    batch_end = min(i + item_batch_size, n_items)
                    batch_items = item_features[i:batch_end, :]
                    batch_preds = self._predict_unbatched(
                        user_features, batch_items
                    )
                    all_preds.append(batch_preds)
                return torch.cat(all_preds, dim=1)
            else:
                # Single batch prediction
                return self._predict_unbatched(
                    user_features, item_features
                )

    def _predict_unbatched(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix],
        item_features: Union[torch.Tensor, sp.spmatrix],
    ) -> torch.Tensor:
        """Internal unbatched prediction logic."""
        # Get latent representations
        user_latent = self._safe_sparse_mm(
            user_features, self.user_latent.weight
        )
        item_latent = self._safe_sparse_mm(
            item_features, self.item_latent.weight
        )

        # Apply activation if specified
        if self.activation is not None:
            user_latent = self.activation(user_latent)
            item_latent = self.activation(item_latent)

        # Compute predictions matrix via latent interactions
        predictions = user_latent @ item_latent.t()

        # Add item biases
        item_bias = self._safe_sparse_mm(
            item_features, self.item_biases.weight
        ).squeeze(-1)
        predictions = predictions + item_bias.unsqueeze(0)

        # Add optional bias terms
        if self.use_user_bias:
            user_bias = self._safe_sparse_mm(
                user_features, self.user_biases.weight
            ).squeeze(-1)
            predictions = predictions + user_bias.unsqueeze(1)

        if self.use_global_bias:
            predictions = predictions + self.global_bias

        return predictions

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the learned user and item embeddings.

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        return (
            self.user_latent.weight.data.clone(),
            self.item_latent.weight.data.clone(),
        )

    def get_user_embedding(
        self,
        user_features: Union[torch.Tensor, sp.spmatrix]
    ) -> torch.Tensor:
        """Get embedding for specific users """
        return self._safe_sparse_mm(user_features, self.user_latent.weight)

    def get_item_embedding(
        self,
        item_features: Union[torch.Tensor, sp.spmatrix]
    ) -> torch.Tensor:
        """Get embedding for specific items"""
        return self._safe_sparse_mm(item_features, self.item_latent.weight)


# class HybridMF(nn.Module):
#     """Hybrid Matrix Factorization model with user and item features"""

#     def __init__(
#             self,
#             n_user_features: int,
#             n_item_features: int,
#             n_latent: int,
#             sparse: bool = False,
#             use_global_bias: bool = True,
#             use_user_bias: bool = False
#     ) -> None:
#         """Initialize hybrid matrix factorization model."""
#         super().__init__()
#         self.n_latent = n_latent
#         self.n_user_features = n_user_features
#         self.n_item_features = n_item_features

#         # Define embeddings for users and items
#         self.user_latent = nn.Embedding(
#             num_embeddings=n_user_features,
#             embedding_dim=n_latent,
#             sparse=sparse
#         )
#         self.item_latent = nn.Embedding(
#             num_embeddings=n_item_features,
#             embedding_dim=n_latent,
#             sparse=sparse
#         )
#         self.item_biases = nn.Embedding(
#             num_embeddings=n_item_features,
#             embedding_dim=1,
#             sparse=sparse
#         )

#         # Optional bias terms
#         self.use_user_bias = use_user_bias
#         self.use_global_bias = use_global_bias

#         if use_user_bias:
#             self.user_biases = nn.Embedding(
#                 num_embeddings=n_user_features,
#                 embedding_dim=1,
#                 sparse=sparse
#             )

#         if use_global_bias:
#             self.global_bias = nn.Parameter(torch.zeros(1))

#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self) -> None:
#         """Initialize model weights and biases with appropriate values."""
#         # Initialize latent factors and biases
#         nn.init.normal_(self.user_latent.weight.data, 0, 1.0/self.n_latent)
#         nn.init.normal_(self.item_latent.weight.data, 0, 1.0/self.n_latent)
#         nn.init.zeros_(self.item_biases.weight.data)

#         if self.use_user_bias:
#             nn.init.zeros_(self.user_biases.weight.data)

#     def forward(
#             self,
#             user_features: torch.Tensor,
#             item_features: torch.Tensor
#     ) -> torch.Tensor:
#         """Predict rating/preference score for user-item pairs."""
#         # Extract latent vectors and compute interactions
#         user_latent = torch.sparse.mm(user_features, self.user_latent.weight)
#         item_latent = torch.sparse.mm(item_features, self.item_latent.weight)
#         # print(user_latent.shape, item_latent.shape)
#         latent_interaction = torch.mul(user_latent, item_latent).sum(dim=-1)

#         # Add all bias terms
#         item_bias = torch.sparse.mm(item_features, self.item_biases.weight)
#         prediction = latent_interaction + item_bias.squeeze()

#         if self.use_user_bias:
#             user_bias = torch.sparse.mm(user_features, self.user_biases.weight)
#             prediction += user_bias.squeeze()

#         if self.use_global_bias:
#             prediction += self.global_bias

#         return prediction

#     def predict(
#             self,
#             user_features: torch.Tensor,
#             item_features: torch.Tensor
#     ) -> torch.Tensor:
#         """Predict ratings for all users and all items at once.

#         Args:
#             user_features: Sparse tensor (n_users, n_user_features)
#             item_features: Sparse tensor (n_items, n_item_features)

#         Returns:
#             Prediction matrix of shape (n_users, n_items)
#         """
#         # Get latent representations
#         user_latent = torch.sparse.mm(
#             user_features, self.user_latent.weight
#         )  # (n_users, n_latent)
#         item_latent = torch.sparse.mm(
#             item_features, self.item_latent.weight
#         )  # (n_items, n_latent)

#         # Compute predictions matrix through latent interactions
#         latent_interaction = user_latent @ item_latent.t()  # (n_users, n_items)

#         # Get item biases and add to predictions (broadcasting)
#         item_bias = torch.sparse.mm(
#             item_features, self.item_biases.weight
#         ).squeeze()  # (n_items)
#         predictions = latent_interaction + item_bias

#         # Apply optional bias terms
#         if self.use_user_bias:
#             user_bias = torch.sparse.mm(
#                 user_features, self.user_biases.weight
#             ).squeeze()  # (n_users)
#             predictions = predictions + user_bias.unsqueeze(1)

#         if self.use_global_bias:
#             predictions = predictions + self.global_bias

#         return predictions

#     def get_top_k_item_features(
#         self,
#         user_feature_idx: int,
#         k: int = 10
#     ) -> Tuple[list, list]:
#         """Get top-k item features for a given user.

#         Args:
#             user_idx: Index of the user feature
#             k: Number of top item features to return

#         Returns:
#             Tuple of (top scores as list, top indices as list)
#         """
#         # Get user latent vector and compute similarity scores
#         user_latent = self.user_latent.weight[user_feature_idx].unsqueeze(0)
#         scores = torch.matmul(user_latent, self.item_latent.weight.T).squeeze()

#         # Add biases
#         scores += self.item_biases.weight.squeeze()

#         if self.use_global_bias:
#             scores += self.global_bias

#         if self.use_user_bias:
#             scores += self.user_biases.weight[user_feature_idx]

#         # Get top-k scores and indices
#         k = min(k, self.n_item_features)
#         top_scores, top_indices = torch.topk(scores, k=k)

#         # Convert tensors to lists
#         top_scores_list = top_scores.detach().cpu().tolist()
#         top_indices_list = top_indices.detach().cpu().tolist()

#         return top_scores_list, top_indices_list

#     # def get_top_k_recommendations(
#     #         self,
#     #         user_features: torch.Tensor,
#     #         item_features: torch.Tensor,
#     #         k: int = 10
#     # ) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     """Get top-k item recommendations for given users."""
#     #     scores = self.forward(user_features, item_features)
#     #     top_scores, top_indices = torch.topk(scores, k=k)
#     #     return top_scores, top_indices

#     # def forward(
#     #         self,
#     #         user_features: torch.Tensor,
#     #         item_features_pos: torch.Tensor,
#     #         item_features_neg: torch.Tensor
#     # ) -> torch.Tensor:
#     #     """Forward pass computing BPR pairwise preference scores."""
#     #     # Compute scores and return difference for BPR optimization
#     #     r_ui = self.predict_score(user_features, item_features_pos)
#     #     r_uj = self.predict_score(user_features, item_features_neg)
#     #     return r_ui - r_uj

# # """
# # Base class for defining matrix factorization model

# # Author: Rimple Sandhu
# # Email: rimple.sandhu@outlook.com
# # """

# # import torch

# # # pylint: disable=invalid-name


# # class HybridMF(torch.nn.Module):
# #     def __init__(
# #             self,
# #             n_user_features: int,
# #             n_item_features: int,
# #             n_latent: int,
# #             sparse: bool = False
# #     ):
# #         # initiate the user and item latent matrices
# #         super(HybridMF, self).__init__()
# #         self.n_latent = n_latent
# #         # user related
# #         self.user_latent = torch.nn.Embedding(
# #             num_embeddings=n_user_features,
# #             embedding_dim=n_latent,
# #             sparse=sparse
# #         )
# #         # self.user_biases = torch.nn.Embedding(
# #         #     num_embeddings=n_user_features,
# #         #     embedding_dim=1,
# #         #     sparse=sparse
# #         # )

# #         # item related
# #         self.item_latent = torch.nn.Embedding(
# #             num_embeddings=n_item_features,
# #             embedding_dim=n_latent,
# #             sparse=sparse
# #         )
# #         self.item_biases = torch.nn.Embedding(
# #             num_embeddings=n_item_features,
# #             embedding_dim=1,
# #             sparse=sparse
# #         )

# #         # initiate
# #         self.initiate()

# #     def initiate(self):
# #         """Initiate user and item latent/bias terms"""
# #         # scale the weights of user/item latent matrix
# #         self.user_latent.weight.data.normal_(0, 1/self.n_latent)
# #         self.item_latent.weight.data.normal_(0, 1/self.n_latent)

# #         # initialize bias to zeros
# #         # self.user_biases.weight.data.zero_()
# #         self.item_biases.weight.data.zero_()

# #     def forward(
# #             self,
# #             Fuser: torch.Tensor,
# #             Fitem_pos: torch.Tensor,
# #             Fitem_neg: torch.Tensor
# #     ):
# #         """Forward simulation"""
# #         # extract the latent vectors for the given features indices
# #         # TODO: Test just iteractions here and return rui (pos) or -ruj(neg)

# #         # latent vectors
# #         Pu = torch.sparse.mm(Fuser, self.user_latent.weight)
# #         Qi_pos = torch.sparse.mm(Fitem_pos, self.item_latent.weight)
# #         Qi_neg = torch.sparse.mm(Fitem_neg, self.item_latent.weight)

# #         # bias vectors
# #         # Bu = torch.sparse.mm(Fuser, self.user_biases.weight)
# #         Bi_pos = torch.sparse.mm(Fitem_pos, self.item_biases.weight)
# #         Bi_neg = torch.sparse.mm(Fitem_neg, self.item_biases.weight)

# #         r_ui = torch.mul(Pu, Qi_pos).sum(dim=-1) + Bi_pos.squeeze()
# #         r_uj = torch.mul(Pu, Qi_neg).sum(dim=-1) + Bi_neg.squeeze()
# #         return r_ui - r_uj

# #     def forward_v2(
# #             self,
# #             user_feature_indices: torch.IntTensor,
# #             item_feature_indices: torch.IntTensor,
# #             user_feature_weights: torch.FloatTensor,
# #             item_feature_weights: torch.FloatTensor
# #     ):
# #         """Forward simulation"""
# #         # extract the latent vectors for the given features indices
# #         p_u = self.user_latent(user_feature_indices)
# #         p_u = p_u*user_feature_weights[..., None]
# #         p_u = p_u.sum(dim=1)

# #         q_i = self.item_latent(item_feature_indices)
# #         q_i = q_i*item_feature_weights[..., None]
# #         q_i = q_i.sum(dim=1)

# #         # extract the biases
# #         b_u = self.user_biases(user_feature_indices)
# #         b_u = b_u*user_feature_weights[..., None]
# #         b_u = b_u.sum(dim=1).squeeze()

# #         b_i = self.item_biases(item_feature_indices)
# #         b_i = b_i*item_feature_weights[..., None]
# #         b_i = b_i.sum(dim=1).squeeze()
# #         # print(p_u.shape, q_i.shape, b_u.shape, b_i.shape)

# #         return torch.mul(p_u, q_i).sum(dim=-1) + b_u + b_i

# #     # def prob_i_over_j(
# #     #     self,
# #     #     pos_user_feature_indices: list[int],
# #     #     pos_item_feature_indices: list[int],
# #     #     neg_user_feature_indices: list[int],
# #     #     neg_item_feature_indices: list[int],
# #     #     # user_feature_weights: list[float] | None = None,
# #     #     # item_feature_weights: list[float] | None = None
# #     # ):
# #     #     """Forward simulation"""
# #     #     # get user/item indices in tensor form

# #     #     pos_r_ui = self.forward_rui(
# #     #         pos_user_feature_indices,
# #     #         pos_item_feature_indices
# #     #     )

# #     #     neg_r_uj = self.forward_rui(
# #     #         neg_user_feature_indices,
# #     #         neg_item_feature_indices
# #     #     )

# #     #     return (1-torch.sigmoid(pos_r_ui-neg_r_uj))
