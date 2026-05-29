import numpy as np

def lu_decomposition(A):
    """Returns: tuple (L, U) where A = L @ U."""
    A = np.array(A, dtype=float) 
    n = len(A)
    L = np.eye(n)
    for i in range(n-1):
        # Check for zero pivots to prevent crashes
        if A[i][i] == 0:
            raise ValueError("Zero pivot encountered. Row swapping (pivoting) is required.")
            
        pivot_row = A[i]
        for j in range(i+1, n):
            working_row = A[j]
            mult = A[j][i] / A[i][i]
            L[j][i] = mult
            working_row[i:] -= mult * pivot_row[i:]
    return L,A