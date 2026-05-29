import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    def leng(v):
        return float(np.matmul(v,v)**0.5)
    return (float(np.matmul(a,b))/(float(np.matmul(a,a)**0.5)*float(np.matmul(b,b)**0.5)) if (leng(a) and leng(b)) else 0)