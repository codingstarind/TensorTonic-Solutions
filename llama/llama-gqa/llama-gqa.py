import torch
import torch.nn.functional as F
import math

def grouped_query_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor,
                            W_v: torch.Tensor, W_o: torch.Tensor,
                            n_heads: int, n_kv_heads: int) -> torch.Tensor:
    """
    Returns: (batch, seq_len, d_model)
    """
    q = x@W_q.T
    k=x@W_k.T
    v=x@W_v.T
    b,s,d = q.shape
    d_head = d // n_heads
    n_rep = n_heads // n_kv_heads
    q = q.reshape(b, s, n_heads, d_head).transpose(1,2)
    k = k.reshape(b, s, n_kv_heads, d_head).transpose(1,2)
    v = v.reshape(b, s, n_kv_heads, d_head).transpose(1,2)
    k_stack = k[:,:,None,:,:].expand(b,n_kv_heads,n_rep,s,d_head).reshape(b,n_heads,s,d_head)
    v_stack = v[:,:,None,:,:].expand(b,n_kv_heads,n_rep,s,d_head).reshape(b,n_heads,s,d_head)
    attn = q@k_stack.transpose(-2,-1)/math.sqrt(d_head)
    attn = F.softmax(attn, dim=-1)
    out = attn@v_stack
    out = out.transpose(1,2).reshape(b,s,d)
    out = out@W_o.T


    return out