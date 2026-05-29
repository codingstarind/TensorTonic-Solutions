import numpy as np

def svd(A):
    """
    Returns: tuple (U, s, Vt) where A = U @ diag(s) @ Vt.
    """
    return np.linalg.svd(np.array(A, dtype=float), full_matrices=False)
    