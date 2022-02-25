# global
from typing import Union, Optional, Tuple, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def roll(x: Union[ivy.Array, ivy.NativeArray],\
    shift: Union[int, Tuple[int]],\
    axis: Optional[Union[int, Tuple[int]]]=None)\
     -> Union[ivy.Array, ivy.NativeArray]:
    """
    Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position. Array elements that roll beyond the first position are re-introduced at the last position.

    :param x: input array.
    :param shift: number of places by which the elements are shifted. If shift is a tuple, then axis must be a tuple of the same size, and each of the given axes must be shifted by the corresponding element in shift. If shift is an int and axis a tuple, then the same shift must be used for all specified axes. If a shift is positive, then array elements must be shifted positively (toward larger indices) along the dimension of axis. If a shift is negative, then array elements must be shifted negatively (toward smaller indices) along the dimension of axis.
    :param axis: axis (or axes) along which elements to shift. If axis is None, the array must be flattened, shifted, and then restored to its original shape. Default: None.
    :return: an output array having the same data type as x and whose elements, relative to x, are shifted.
    """
    return _cur_framework(x).roll(x, shift, axis)


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
