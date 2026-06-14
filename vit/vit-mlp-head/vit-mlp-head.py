import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE
    h_cls = encoder_output[:,0,:]
    b, d = h_cls.shape
    mean = np.mean(h_cls, axis=-1, keepdims=True)
    var = np.var(h_cls, axis=-1, keepdims=True)
    h_cls_cap = (h_cls-mean)/(var**0.5+1e-6)
    if W_head is None:
        W_head = np.random.randn(d, num_classes)*0.02
    logits = h_cls_cap@W_head
    return logits