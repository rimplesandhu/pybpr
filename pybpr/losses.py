from typing import Protocol

import torch


class PairwiseLossFn(Protocol):
    """Protocol for pairwise ranking loss functions.

    Expected signature: (pos_scores, neg_scores) -> loss_tensor

    Example - Custom loss:
        def my_loss(pos, neg):
            return torch.mean(torch.relu(1.0 - (pos - neg)))

    Example - Adapting PyTorch loss (MarginRankingLoss):
        margin_loss = torch.nn.MarginRankingLoss(margin=1.0)
        def adapted_loss(pos, neg):
            target = torch.ones_like(pos)
            return margin_loss(pos, neg, target)
    """
    def __call__(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        ...


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    sigma: float = 1.0
) -> torch.Tensor:
    """Compute Bayesian Personalized Ranking (BPR) loss."""
    # BPR: negative log-sigmoid of difference
    # TODO: Proper scoring rule, briar score, log score
    # TODO: difference = pos_wgt*positive - neg_wgt*negative
    difference = positive_scores - negative_scores
    loss = -torch.log(torch.sigmoid(sigma * difference))
    return loss.mean()


def bpr_loss_v2(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Compute BPR loss (alternative formulation)."""
    # BPR: separate sigmoid terms for positive and negative
    loss = -torch.log(torch.sigmoid(positive_scores))
    loss += -torch.log(torch.sigmoid(-negative_scores))
    return loss.mean()


def hinge_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """Compute Hinge Loss (Margin Ranking Loss)."""
    # Calculate max(0, margin - (positive - negative))
    difference = positive_scores - negative_scores
    loss = torch.nn.functional.relu(margin - difference).mean()
    return loss


def warp_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    num_items: int
) -> torch.Tensor:
    """Compute WARP loss (simplified version)."""
    # Compute rank approximation
    diff = positive_scores - negative_scores

    # Count how many negatives score higher than positives
    rank = torch.sum(diff < 0).float() + 1

    # Log weight based on rank
    weight = (
        torch.log(rank + 1) /
        torch.log(torch.tensor(num_items, dtype=torch.float))
    )

    # Hinge loss weighted by rank
    margin_loss = torch.nn.functional.relu(1 - diff)
    loss = (weight * margin_loss).mean()

    return loss


__all__ = [
    'PairwiseLossFn',
    'bpr_loss',
    'bpr_loss_v2',
    'hinge_loss',
    'warp_loss',
]
