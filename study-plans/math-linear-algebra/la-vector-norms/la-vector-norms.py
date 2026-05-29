import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    man = float(np.sum(np.abs(v)))
    euc = float(np.matmul(v,v)**0.5)
    linf = np.max(np.abs(v))
    return np.array([man,euc,linf])