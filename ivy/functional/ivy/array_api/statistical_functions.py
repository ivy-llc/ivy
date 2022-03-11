# global
from typing import Union, Tuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def min(x: Union[ivy.Array, ivy.NativeArray],
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False) \
        -> ivy.Array:
    """
    Return the minimum value of the input array x.

    :param x: Input array containing elements to min.
    :param axis: Axis or axes along which minimum values must be computed, default is None.
    :param keepdims, optional axis or axes along which minimum values must be computed, default is None.
    :param f: Machine learning framework. Inferred from inputs if None.
    :return: array containing minimum value.
    """
    return _cur_framework.min(x, axis, keepdims)
