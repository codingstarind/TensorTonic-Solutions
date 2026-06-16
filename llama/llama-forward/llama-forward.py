import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def llama_forward(token_ids, W_embed, blocks, rms_final, W_head, freqs_cos, freqs_sin, eps=1e-6):
    """
    Returns: logits tensor (batch, seq_len, vocab_size) from Llama 3 forward pass.
    """
    def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Returns: Normalized tensor of same shape as x
        """
        # YOUR CODE HERE
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x * gamma / rms
    def llama_block(x, rms_w1, rms_w2, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, W_gate, W_up, W_down, freqs_cos, freqs_sin, eps=1e-6):
        """
        Returns: dict with key "output" containing tensor (batch, seq_len, d_model) as nested list, rounded to 4 decimals.
        """
        # YOUR CODE HERE
        
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
        def apply_rope(q, k, freqs_cos, freqs_sin):
            """
            Returns: tuple of (q_rotated, k_rotated) same shapes as input
            """
            # YOUR CODE HERE
            s = q.shape[-2]             # actual sequence length
            freqs_cos = freqs_cos[:s]   # (max_seq_len, d//2) → (s, d//2)
            freqs_sin = freqs_sin[:s]
        
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
            q,k_stack = apply_rope(q,k_stack, freqs_cos,freqs_sin)
            attn = q@k_stack.transpose(-2,-1)/math.sqrt(d_head)
            mask = torch.tril(torch.ones(s, s, dtype=torch.float32))
            attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = attn@v_stack
            out = out.transpose(1,2).reshape(b,s,d)
            out = out@W_o.T
            return out
        def swiglu_ffn(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
            """
            Apply SwiGLU feed-forward network.
            """
            # Your code here
            swish = nn.SiLU()
            return (swish(x@W_gate.T)*(x@W_up.T))@W_down.T
       
        rms_w1 = torch.tensor(rms_w1)
        rms_w2 = torch.tensor(rms_w2)
        W_q = torch.tensor(W_q)
        W_k = torch.tensor(W_k)
        W_v = torch.tensor(W_v)
        W_o = torch.tensor(W_o)
        freqs_cos=torch.tensor(freqs_cos)
        freqs_sin=torch.tensor(freqs_sin)
        W_gate = torch.tensor(W_gate)
        W_up = torch.tensor(W_up)
        W_down = torch.tensor(W_down)
        x_cap = rms_norm(x,rms_w1,eps)
        attn_output = grouped_query_attention(x_cap,W_q,W_k,W_v,W_o,n_heads,n_kv_heads)
        h = x + attn_output
        h_cap = rms_norm(h,rms_w2,eps)
        ffn = swiglu_ffn(h_cap, W_gate,W_up,W_down)
        output = h+ffn
      
        return output
    h = torch.tensor(W_embed[token_ids])
    rms_final =torch.tensor(rms_final)
    W_head=torch.tensor(W_head)
    for block in blocks:
        rms_w1 = block["rms_w1"]
        rms_w2 = block["rms_w2"]
        W_q = block["W_q"]
        W_k = block["W_k"]
        W_v = block["W_v"]
        W_o = block["W_o"]
        n_heads = block["n_heads"]
        n_kv_heads = block["n_kv_heads"]
        W_gate = block["W_gate"]
        W_up = block["W_up"]
        W_down = block["W_down"]
        h = llama_block(h,rms_w1,rms_w2,W_q,W_k,W_v,W_o,n_heads,n_kv_heads,W_gate,W_up,W_down,freqs_cos,freqs_sin,eps)
    h_f = rms_norm(h, rms_final,eps)
    return h_f@W_head.T
        
    
        