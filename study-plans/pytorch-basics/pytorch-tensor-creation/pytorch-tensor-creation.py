import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == "zeros":
        return torch.zeros(shape)
    elif method == "ones":
        return torch.ones(shape)
    else:
        return torch.full(shape, value)