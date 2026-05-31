import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    b, h, w, c = image.shape
    p = patch_size
    n = (h//p) * (w//p)
    d = embed_dim
    ppc = p*p*c
    patches = image.reshape(b, h // patch_size, patch_size, w // patch_size, patch_size,c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(b, n, ppc)

    if W_proj is None:
        W_proj = np.random.randn(ppc,d)*0.02
    else:
        W_proj = np.array(W_proj)
    lin_proj_image = patches @ W_proj
    
    return lin_proj_image