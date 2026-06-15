import torch

def apply_rope(q, k, freqs_cos, freqs_sin):
    """
    Returns: tuple of (q_rotated, k_rotated) same shapes as input
    """
    # YOUR CODE HERE
    q_x = q[..., 0::2]
    q_y = q[..., 1::2]
    k_x = k[..., 0::2]
    k_y = k[..., 1::2]
    q_stack = torch.stack((q_x,q_y),dim=0)*freqs_cos + torch.stack((-q_y,q_x),dim=0)*freqs_sin
    q_rotated = torch.zeros_like(q)
    q_rotated[..., 0::2] = q_stack[0]
    q_rotated[..., 1::2] = q_stack[1]
    k_stack = torch.stack((k_x,k_y),dim=0)*freqs_cos + torch.stack((-k_y,k_x),dim=0)*freqs_sin
    k_rotated = torch.zeros_like(k)
    k_rotated[..., 0::2] = k_stack[0]
    k_rotated[..., 1::2] = k_stack[1]
    return q_rotated, k_rotated