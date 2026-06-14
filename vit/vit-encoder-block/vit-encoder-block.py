import numpy as np
from scipy.special import softmax, erf

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE
    b, seq_len,embed = x.shape
    mean =x.mean(axis=-1).reshape(b,-1,1)
    var = x.var(axis=-1).reshape(b,-1,1)
    layer_normed_x = (x-mean)/(var**0.5)
    q = layer_normed_x @ Wq #(b, seq, em) * (em, em)
    k = layer_normed_x @ Wk
    v = layer_normed_x @ Wv
    b, seq_len,embed  = q.shape
    dk = embed//num_heads
    q=q.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
    k=k.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
    v=v.reshape(b,seq_len,num_heads, dk).transpose(0,2,1,3) 
    y = (q@k.transpose(0,1,3,2))/(dk**0.5)
    msa = ((softmax(y, axis=-1) @ v).transpose(0,2,1,3).reshape(b,seq_len,embed))@Wo #(b,num_heads,seq_len,dk) -> (b, seq_len, embed)
    
    x_f = x+msa
    mean =x_f.mean(axis=-1).reshape(b,-1,1)
    var = x_f.var(axis=-1).reshape(b,-1,1)
    layer_normed_x_f = (x_f-mean)/(var**0.5)
    def gelu_exact(x):
        """Exact GELU activation function using the error function."""
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    gelu_inner = layer_normed_x_f@W1
    mlp = gelu_exact(gelu_inner)@W2
    
    return x_f+mlp