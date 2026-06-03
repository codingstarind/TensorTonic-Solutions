import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: Normalized tensor of same shape as x
    """
    deno = torch.sqrt(torch.mean(torch.square(x), dim=-1)+eps)
    return torch.multiply(x, gamma)/ deno.unsqueeze(-1)
    