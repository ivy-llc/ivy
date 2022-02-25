# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def argsort(x: Union[ivy.Array, ivy.NativeArray],
            axis: int = -1,
            descending: bool = False,
            stable: bool = True,
            f: ivy.Framework = None)\
            -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the indices of a tensor that give its sorted order along an axis.

    :param x: Input array containing elements to argsort.
    :type x: array
    :param axis: Axis to perform the argsort, default is -1.
    :type axis: int, optional
    :param descending: Sort order, default is False.
    :type descending: bool, optional
    :param stable: Sort stability, default is True.
    :type stable: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The indices that would sort each slice of the given values along the given axis.
    """
    return _cur_framework(x, f=f).argsort(x, axis, descending, stable)
