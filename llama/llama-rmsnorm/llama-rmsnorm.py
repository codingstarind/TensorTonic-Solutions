import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: Normalized tensor of same shape as x
    """
    # YOUR CODE HERE
    return (x/(torch.mean(x**2, dim=-1)+eps).unsqueeze(-1)**0.5)*gamma