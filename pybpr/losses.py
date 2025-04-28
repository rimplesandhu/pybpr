import torch
import torch.nn.functional as F


def bpr_loss(positive_ratings, negative_ratings, sigma=1.0):
    """
    Compute Bayesian Personalized Ranking (BPR) loss.

    Parameters:
    -----------
    positive_ratings : torch.Tensor
        Predicted ratings for positive interactions
    negative_ratings : torch.Tensor
        Predicted ratings for negative interactions
    sigma : float, optional
        Scaling factor for the sigmoid function, default is 1.0

    Returns:
    --------
    loss : torch.Tensor
        The computed BPR loss
    """
    # BPR loss is the negative log of the sigmoid of the difference
    # TODO: Proper scoring rule, briar score, log score
    # TODO: difference = pos_wgt*positive_ratings - neg_wgt*negative_ratings
    difference = positive_ratings - negative_ratings
    loss = -torch.log(torch.sigmoid(sigma * difference))
    return loss.mean()


def bpr_loss_v2(positive_ratings, negative_ratings, sigma=1.0):
    """
    Compute Bayesian Personalized Ranking (BPR) loss.

    Parameters:
    -----------
    positive_ratings : torch.Tensor
        Predicted ratings for positive interactions
    negative_ratings : torch.Tensor
        Predicted ratings for negative interactions
    sigma : float, optional
        Scaling factor for the sigmoid function, default is 1.0

    Returns:
    --------
    loss : torch.Tensor
        The computed BPR loss
    """
    # BPR loss is the negative log of the sigmoid of the difference
    loss = -torch.log(torch.sigmoid(positive_ratings))
    loss += -torch.log(torch.sigmoid(-negative_ratings))
    return loss.mean()


def hinge_loss(positive_ratings, negative_ratings, margin=1.0):
    """
    Compute Hinge Loss (Margin Ranking Loss) for pairwise ranking.

    Parameters:
    -----------
    positive_ratings : torch.Tensor
        Predicted ratings for positive interactions
    negative_ratings : torch.Tensor
        Predicted ratings for negative interactions
    margin : float, optional
        The margin between positive and negative samples

    Returns:
    --------
    loss : torch.Tensor
        The computed hinge loss
    """
    # Calculate max(0, margin - (positive - negative))
    difference = positive_ratings - negative_ratings
    loss = torch.nn.functional.relu(margin - difference).mean()
    return loss


def warp_loss(positive_ratings, negative_ratings, num_items):
    """
    A simplified version of WARP loss concept.

    Parameters:
    -----------
    positive_ratings : torch.Tensor
        Predicted ratings for positive interactions
    negative_ratings : torch.Tensor
        Predicted ratings for negative interactions
    num_items : int
        Total number of items in the dataset

    Returns:
    --------
    loss : torch.Tensor
        The computed WARP loss (simplified)
    """
    # Compute rank approximation
    diff = positive_ratings - negative_ratings
    # How many negatives score higher than positives
    rank = torch.sum(diff < 0).float() + 1

    # Log weight of the rank
    weight = torch.log(rank + 1) / \
        torch.log(torch.tensor(num_items, dtype=torch.float))

    # Compute hinge loss weighted by rank
    margin_loss = torch.nn.functional.relu(1 - diff)
    loss = (weight * margin_loss).mean()

    return loss
