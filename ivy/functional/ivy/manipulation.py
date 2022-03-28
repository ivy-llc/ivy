# global
from typing import Union, Optional, Tuple, List, Iterable
from numbers import Number
# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def squeeze(x: Union[ivy.Array, ivy.NativeArray],
            axis: Union[int, Tuple[int, ...]])\
        -> ivy.Array:
    """
    Removes singleton dimensions (axes) from ``x``.
    Parameters
    ----------
    x: array
        input array.
    axis: Union[int, Tuple[int, ...]]
        axis (or axes) to squeeze. If a specified axis has a size greater than one, a ``ValueError`` must be raised.
    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.
    """
    return _cur_framework(x).squeeze(x, axis)

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


def constant_pad(x: Union[ivy.Array, ivy.NativeArray], pad_width: Iterable[Tuple[int]], value: Number = 0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Pads an array with a constant value.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :param value: The constant value to pad the array with.
    :type value: float or int, default zero
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _cur_framework(x).constant_pad(x, pad_width, value)


def zero_pad(x: Union[ivy.Array, ivy.NativeArray], pad_width: Iterable[Tuple[int]])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Pads an array with zeros.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _cur_framework(x).zero_pad(x, pad_width)


def swapaxes(x: Union[ivy.Array, ivy.NativeArray], axis0: int, axis1: int)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Interchange two axes of an array.

    :param x: Input array.
    :type x: array
    :param axis0: First axis to be swapped.
    :type axis0: int
    :param axis1: Second axis to be swapped.
    :type axis1: int
    :return: x with its axes permuted.
    """
    return _cur_framework(x).swapaxes(x, axis0, axis1)


def clip(x: Union[ivy.Array, ivy.NativeArray], x_min: Union[Number, Union[ivy.Array, ivy.NativeArray]],
         x_max: Union[Number, Union[ivy.Array, ivy.NativeArray]])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges (element-wise).
    For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    :param x: Input array containing elements to clip.
    :type x: array
    :param x_min: Minimum value.
    :type x_min: scalar or array
    :param x_max: Maximum value.
    :type x_max: scalar or array
    :return: An array with the elements of x, but where values < x_min are replaced with x_min,
                and those > x_max with x_max.
    """
    return _cur_framework(x).clip(x, x_min, x_max)