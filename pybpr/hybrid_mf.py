#!/usr/bin/env python
"""
Base class for defining matrix factorization model for recommendation systems.

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

from typing import Tuple
import torch
import torch.nn as nn


class HybridMF(nn.Module):
    """Hybrid Matrix Factorization model with user and item features"""

    def __init__(
            self,
            n_user_features: int,
            n_item_features: int,
            n_latent: int,
            sparse: bool = False,
            use_global_bias: bool = True,
            use_user_bias: bool = False
    ) -> None:
        """Initialize hybrid matrix factorization model."""
        super().__init__()
        self.n_latent = n_latent
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features

        # Define embeddings for users and items
        self.user_latent = nn.Embedding(
            num_embeddings=n_user_features,
            embedding_dim=n_latent,
            sparse=sparse
        )
        self.item_latent = nn.Embedding(
            num_embeddings=n_item_features,
            embedding_dim=n_latent,
            sparse=sparse
        )
        self.item_biases = nn.Embedding(
            num_embeddings=n_item_features,
            embedding_dim=1,
            sparse=sparse
        )

        # Optional bias terms
        self.use_user_bias = use_user_bias
        self.use_global_bias = use_global_bias

        if use_user_bias:
            self.user_biases = nn.Embedding(
                num_embeddings=n_user_features,
                embedding_dim=1,
                sparse=sparse
            )

        if use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights and biases with appropriate values."""
        # Initialize latent factors and biases
        nn.init.normal_(self.user_latent.weight.data, 0, 1.0/self.n_latent)
        nn.init.normal_(self.item_latent.weight.data, 0, 1.0/self.n_latent)
        nn.init.zeros_(self.item_biases.weight.data)

        if self.use_user_bias:
            nn.init.zeros_(self.user_biases.weight.data)

    def forward(
            self,
            user_features: torch.Tensor,
            item_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict rating/preference score for user-item pairs."""
        # Extract latent vectors and compute interactions
        user_latent = torch.sparse.mm(user_features, self.user_latent.weight)
        item_latent = torch.sparse.mm(item_features, self.item_latent.weight)
        latent_interaction = torch.mul(user_latent, item_latent).sum(dim=-1)

        # Add all bias terms
        item_bias = torch.sparse.mm(item_features, self.item_biases.weight)
        prediction = latent_interaction + item_bias.squeeze()

        if self.use_user_bias:
            user_bias = torch.sparse.mm(user_features, self.user_biases.weight)
            prediction += user_bias.squeeze()

        if self.use_global_bias:
            prediction += self.global_bias

        return prediction

    def get_top_k_item_features(
        self,
        user_feature_idx: int,
        k: int = 10
    ) -> Tuple[list, list]:
        """Get top-k item features for a given user.

        Args:
            user_idx: Index of the user feature
            k: Number of top item features to return

        Returns:
            Tuple of (top scores as list, top indices as list)
        """
        # Get user latent vector and compute similarity scores
        user_latent = self.user_latent.weight[user_feature_idx].unsqueeze(0)
        scores = torch.matmul(user_latent, self.item_latent.weight.T).squeeze()

        # Add biases
        scores += self.item_biases.weight.squeeze()

        if self.use_global_bias:
            scores += self.global_bias

        if self.use_user_bias:
            scores += self.user_biases.weight[user_feature_idx]

        # Get top-k scores and indices
        k = min(k, self.n_item_features)
        top_scores, top_indices = torch.topk(scores, k=k)

        # Convert tensors to lists
        top_scores_list = top_scores.detach().cpu().tolist()
        top_indices_list = top_indices.detach().cpu().tolist()

        return top_scores_list, top_indices_list

    # def get_top_k_recommendations(
    #         self,
    #         user_features: torch.Tensor,
    #         item_features: torch.Tensor,
    #         k: int = 10
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Get top-k item recommendations for given users."""
    #     scores = self.forward(user_features, item_features)
    #     top_scores, top_indices = torch.topk(scores, k=k)
    #     return top_scores, top_indices

    # def forward(
    #         self,
    #         user_features: torch.Tensor,
    #         item_features_pos: torch.Tensor,
    #         item_features_neg: torch.Tensor
    # ) -> torch.Tensor:
    #     """Forward pass computing BPR pairwise preference scores."""
    #     # Compute scores and return difference for BPR optimization
    #     r_ui = self.predict_score(user_features, item_features_pos)
    #     r_uj = self.predict_score(user_features, item_features_neg)
    #     return r_ui - r_uj

# """
# Base class for defining matrix factorization model

# Author: Rimple Sandhu
# Email: rimple.sandhu@outlook.com
# """

# import torch

# # pylint: disable=invalid-name


# class HybridMF(torch.nn.Module):
#     def __init__(
#             self,
#             n_user_features: int,
#             n_item_features: int,
#             n_latent: int,
#             sparse: bool = False
#     ):
#         # initiate the user and item latent matrices
#         super(HybridMF, self).__init__()
#         self.n_latent = n_latent
#         # user related
#         self.user_latent = torch.nn.Embedding(
#             num_embeddings=n_user_features,
#             embedding_dim=n_latent,
#             sparse=sparse
#         )
#         # self.user_biases = torch.nn.Embedding(
#         #     num_embeddings=n_user_features,
#         #     embedding_dim=1,
#         #     sparse=sparse
#         # )

#         # item related
#         self.item_latent = torch.nn.Embedding(
#             num_embeddings=n_item_features,
#             embedding_dim=n_latent,
#             sparse=sparse
#         )
#         self.item_biases = torch.nn.Embedding(
#             num_embeddings=n_item_features,
#             embedding_dim=1,
#             sparse=sparse
#         )

#         # initiate
#         self.initiate()

#     def initiate(self):
#         """Initiate user and item latent/bias terms"""
#         # scale the weights of user/item latent matrix
#         self.user_latent.weight.data.normal_(0, 1/self.n_latent)
#         self.item_latent.weight.data.normal_(0, 1/self.n_latent)

#         # initialize bias to zeros
#         # self.user_biases.weight.data.zero_()
#         self.item_biases.weight.data.zero_()

#     def forward(
#             self,
#             Fuser: torch.Tensor,
#             Fitem_pos: torch.Tensor,
#             Fitem_neg: torch.Tensor
#     ):
#         """Forward simulation"""
#         # extract the latent vectors for the given features indices
#         # TODO: Test just iteractions here and return rui (pos) or -ruj(neg)

#         # latent vectors
#         Pu = torch.sparse.mm(Fuser, self.user_latent.weight)
#         Qi_pos = torch.sparse.mm(Fitem_pos, self.item_latent.weight)
#         Qi_neg = torch.sparse.mm(Fitem_neg, self.item_latent.weight)

#         # bias vectors
#         # Bu = torch.sparse.mm(Fuser, self.user_biases.weight)
#         Bi_pos = torch.sparse.mm(Fitem_pos, self.item_biases.weight)
#         Bi_neg = torch.sparse.mm(Fitem_neg, self.item_biases.weight)

#         r_ui = torch.mul(Pu, Qi_pos).sum(dim=-1) + Bi_pos.squeeze()
#         r_uj = torch.mul(Pu, Qi_neg).sum(dim=-1) + Bi_neg.squeeze()
#         return r_ui - r_uj

#     def forward_v2(
#             self,
#             user_feature_indices: torch.IntTensor,
#             item_feature_indices: torch.IntTensor,
#             user_feature_weights: torch.FloatTensor,
#             item_feature_weights: torch.FloatTensor
#     ):
#         """Forward simulation"""
#         # extract the latent vectors for the given features indices
#         p_u = self.user_latent(user_feature_indices)
#         p_u = p_u*user_feature_weights[..., None]
#         p_u = p_u.sum(dim=1)

#         q_i = self.item_latent(item_feature_indices)
#         q_i = q_i*item_feature_weights[..., None]
#         q_i = q_i.sum(dim=1)

#         # extract the biases
#         b_u = self.user_biases(user_feature_indices)
#         b_u = b_u*user_feature_weights[..., None]
#         b_u = b_u.sum(dim=1).squeeze()

#         b_i = self.item_biases(item_feature_indices)
#         b_i = b_i*item_feature_weights[..., None]
#         b_i = b_i.sum(dim=1).squeeze()
#         # print(p_u.shape, q_i.shape, b_u.shape, b_i.shape)

#         return torch.mul(p_u, q_i).sum(dim=-1) + b_u + b_i

#     # def prob_i_over_j(
#     #     self,
#     #     pos_user_feature_indices: list[int],
#     #     pos_item_feature_indices: list[int],
#     #     neg_user_feature_indices: list[int],
#     #     neg_item_feature_indices: list[int],
#     #     # user_feature_weights: list[float] | None = None,
#     #     # item_feature_weights: list[float] | None = None
#     # ):
#     #     """Forward simulation"""
#     #     # get user/item indices in tensor form

#     #     pos_r_ui = self.forward_rui(
#     #         pos_user_feature_indices,
#     #         pos_item_feature_indices
#     #     )

#     #     neg_r_uj = self.forward_rui(
#     #         neg_user_feature_indices,
#     #         neg_item_feature_indices
#     #     )

#     #     return (1-torch.sigmoid(pos_r_ui-neg_r_uj))
