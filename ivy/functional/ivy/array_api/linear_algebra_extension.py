# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def det(x: ivy.array) -> ivy.array:
    """
    Computes determinant of array

    :param x: Input array.
    :return : if x is a two-dimensional array, a zero-dimensional array containing the determinant
              otherwise, a non-zero dimensional array containing the determinant for each square matrix.
"""
    return _cur_framework(x).det(x)