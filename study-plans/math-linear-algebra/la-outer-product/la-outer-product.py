import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u=np.array(u).reshape(-1,1)
    print(u.shape)
    v = np.array(v)
    print(v.shape)
    v_T = v.reshape(1,-1)
    print(v_T.shape)
    return(np.matmul(u,v_T))