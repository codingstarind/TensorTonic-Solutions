import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    b, h, w, c = image.shape
    p = patch_size
    n = h//p * w//p
    patches = image.reshape(b, h//p,p,w//p,p,c)
    patches = patches.transpose(0,1,3,2,4,5).reshape(b,n,p*p*c)
    
    patch_dim = p**2 *c
    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim)*0.02
    return patches@W_proj