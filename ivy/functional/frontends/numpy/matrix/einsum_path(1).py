import numpy as np
from numpy.core.einsumfunc import _parse_einsum_input

def matrix_multiply(a, b):
    subscript = 'ij,jk->ik'
    input_list = [a, b]
    path_info = _parse_einsum_input(subscript, *input_list)
    path = np.einsum_path(*path_info.operands, optimize=path_info.optimize)
    result = np.einsum(subscript, *input_list, optimize=path)
    return result
