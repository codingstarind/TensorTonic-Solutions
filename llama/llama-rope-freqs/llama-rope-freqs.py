import torch

def precompute_rope_freqs(max_seq_len, d_head, base=10000.0):
    """
    Returns: tuple of (cos_table, sin_table) both shape (max_seq_len, d_head//2)
    """
        # YOUR CODE HERE
    
    theta = 1.0/(base**(2*torch.arange(start=0,end=d_head//2,step=1,dtype=torch.float32)/d_head))
    p = torch.arange(0,max_seq_len,1,dtype=torch.float32).unsqueeze(1) * theta.unsqueeze(0)
    cos_table =torch.cos(p)
    sin_table = torch.sin(p)
    return cos_table,sin_table
