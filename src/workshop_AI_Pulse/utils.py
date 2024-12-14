import torch


def pairwise_cosine_similarity_matrix(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise cosine similarity matrix for two sets of vectors.

    Parameters
    ----------
    x1 : torch.Tensor
        Tensor of shape (N, D) representing the first set of vectors.
    x2 : torch.Tensor
        Tensor of shape (M, D) representing the second set of vectors.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, M) representing pairwise cosine similarities.
    """
    eps = 1e-10
    x1_norm = x1.norm(dim=-1, keepdim=True).clamp(min=eps)
    x2_norm = x2.norm(dim=-1, keepdim=True).clamp(min=eps)
    x1_scaled = x1 / x1_norm
    x2_scaled = x2 / x2_norm
    return torch.mm(x1_scaled, x2_scaled.transpose(0, 1))
