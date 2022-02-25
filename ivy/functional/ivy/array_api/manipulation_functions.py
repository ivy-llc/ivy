# global
from typing import Union, Optional, Tuple, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def flip(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
        -> ivy.Array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    :param x: input array.
    :param axis: axis (or axes) along which to flip. If axis is None, the function must flip all input array axes.
        If axis is negative, the function must count from the last dimension. If provided more than one axis,
        the function must flip only the specified axes. Default: None.
    :return: an output array having the same data type and shape as x and whose elements, relative to x, are reordered.
    """

    return _cur_framework(x).any(x, axis)


def squeeze(x: ivy.Array,
            axis: Union[int, Tuple[int], List[int]])\
        -> ivy.Array:
    """
    Removes singleton dimensions (axes) from x.

    :param x: input array.
    :param axis: axis (or axes) to squeeze. If a specified axis has a size greater than one, a ValueError must be raised.
    :return: an output array having the same data type and elements as x.
    """

    return _cur_framework(x).squeeze(x, axis)
