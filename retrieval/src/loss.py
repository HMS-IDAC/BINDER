import torch


def _eye_like(tensor):
    """
    Create identity matrix with same size and device as input
    """
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


def hardest_loss(anchor, same, margin=1.0, max_dist=10.0):
    """
    Hardest in batch triplet loss as described in
    hardnet paper: https://arxiv.org/pdf/1610.07940.pdf
    Args:
        anchor (torch.tensor): Embeddings for anchor images.
        same (torch.tensor): Embeddings for same images.
        margin (float): Margin for triplet loss.
        max_dist (float): Maximum distance to add to same pairs, ensuring they won't
            be selected for hard negatives.
    Returns:
        torch.tensor: Loss value.
    """
    dist_matrix = torch.cdist(anchor, same)
    pos = torch.diag(dist_matrix)  # matching images will appear along diagonal

    # add max_down diagonal so matching images are not selected as minimum
    dist_matrix = dist_matrix + _eye_like(dist_matrix) * max_dist

    min_neg = torch.min(torch.min(dist_matrix, 1)[0], torch.min(dist_matrix, 0)[0])

    loss = torch.clamp(pos - min_neg + margin, 0.0)
    loss = torch.mean(loss)

    return loss
