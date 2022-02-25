# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def argsort(x: Union[ivy.Array, ivy.NativeArray],
            axis: int = -1,
            descending: bool = False,
            stable: bool = True)\
            -> ivy.Array:
    """
    Returns the indices of a tensor that give its sorted order along an axis.

    :param x: Input array containing elements to argsort.
    :param axis: Axis to perform the argsort, default is -1.
    :param descending: Sort order, default is False.
    :param stable: Sort stability, default is True.
    :return: The indices that would sort each slice of the given values along the given axis.
    """
    return _cur_framework(x).argsort(x, axis, descending, stable)
