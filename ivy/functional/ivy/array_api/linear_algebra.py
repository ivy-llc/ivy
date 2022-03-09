# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def det(x: ivy.array) \
    -> ivy.array:
    """
    Returns the determinant of a square matrix (or a stack of square matrices) x.

    :param x:  input array having shape (..., M, M) and whose innermost two dimensions form square matrices. Should
               have a floating-point data type.
    :return :  if x is a two-dimensional array, a zero-dimensional array containing the determinant; otherwise, a non-zero
               dimensional array containing the determinant for each square matrix. The returned array must have the same data type as x.
    """
    return _cur_framework(x).det(x)
