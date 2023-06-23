import numpy as np
from numpy.core.einsumfunc import einsum_path

def matrix_multiply(a, b):
    path = einsum_path('ij, jk -> ik', a, b)
    result = np.einsum('ij, jk -> ik', a, b, optimize=path)
    return result
