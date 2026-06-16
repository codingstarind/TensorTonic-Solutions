import torch

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Returns: (batch, n_kv_heads * n_rep, seq_len, d_head)
    """
    # YOUR CODE HERE
    return kv[:,:,None,:,:].expand(kv.shape[0],kv.shape[1],n_rep,kv.shape[2],kv.shape[3]).reshape(kv.shape[0],kv.shape[1]*n_rep,kv.shape[2],kv.shape[3])