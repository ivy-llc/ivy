# global
from typing import Union, Optional, Tuple, List, Iterable

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def flip(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
        -> ivy.Array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    Parameters
    ----------
    x:
        input array.
    axis:
        axis (or axes) along which to flip. If ``axis`` is ``None``, the function must flip all input array axes. If ``axis`` is negative, the function must count from the last dimension. If provided more than one axis, the function must flip only the specified axes. Default: ``None``.

    Returns
    -------
    out:
        an output array having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
    """

    return _cur_framework(x).flip(x, axis)


def expand_dims(x: Union[ivy.Array, ivy.NativeArray],
                axis: Optional[Union[int, Tuple[int], List[int]]] = None) \
        -> ivy.Array:
    """
    Expands the shape of an array.
    Inserts a new axis that will appear at the axis position in the expanded array shape.

    :param x: Input array.
    :type x: array
    :param axis: Position in the expanded axes where the new axis is placed.
    :type axis: int
    :return: array with the number of dimensions increased by onearray
    """
    return _cur_framework(x).expand_dims(x, axis)


def permute_dims(x: Union[ivy.Array, ivy.NativeArray],
                 axes: Tuple[int,...])\
        -> ivy.Array:
    """
    Permutes the axes (dimensions) of an array x.

    Parameters
    ----------
    x:
        input array.
    axes:
        tuple containing a permutation of (0, 1, ..., N-1) where N is the number of axes (dimensions) of x.

    Returns
    -------
    out:
        an array containing the axes permutation. The returned array must have the same data type as x.
    """
    return _cur_framework(x).permute_dims(x, axes)
# Extra #
# ------#


def split(x: Union[ivy.Array, ivy.NativeArray], num_or_size_splits: Union[int, Iterable[int]] = None, axis: int = 0,
          with_remainder: bool = False) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Splits an array into multiple sub-arrays.

    :param x: Tensor to be divided into sub-arrays.
    :type x: array
    :param num_or_size_splits: Number of equal arrays to divide the array into along the given axis if an integer.
                               The size of each split element if a sequence of integers.
                               Default is to divide into as many 1-dimensional arrays as the axis dimension.
    :type num_or_size_splits: int, optional
    :param axis: The axis along which to split, default is 0.
    :type axis: int, optional
    :param with_remainder: If the tensor does not split evenly, then store the last remainder entry. Default is False.
    :type with_remainder: bool, optional
    :return: A list of sub-arrays.
    """
    return _cur_framework(x).split(x, num_or_size_splits, axis, with_remainder)


def repeat(x: Union[ivy.Array, ivy.NativeArray], repeats: Union[int, Iterable[int]], axis: int = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Repeat values along a given dimension

    :param x: Input array.
    :type x: array
    :param repeats: The number of repetitions for each element. repeats is broadcast to fit the shape of the given axis.
    :type repeats: int or sequence of ints.
    :param axis: The axis along which to repeat values.
                  By default, use the flattened input array, and return a flat output array.
    :type axis: int, optional
    :return: The repeated output array.
    """
    return _cur_framework(x).repeat(x, repeats, axis)


def tile(x: Union[ivy.Array, ivy.NativeArray], reps: Iterable[int])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Constructs an array by repeating x the number of times given by reps.

    :param x: Input array.
    :type x: array
    :param reps: The number of repetitions of x along each axis.
    :type reps: sequence of ints
    :return: The tiled output array.
    """
    return _cur_framework(x).tile(x, reps)