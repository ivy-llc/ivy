# global
from typing import Union, Optional, Tuple, List, Iterable
from numbers import Number
# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def roll(x: Union[ivy.Array, ivy.NativeArray],
         shift: Union[int, Tuple[int, ...]],
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         out: Optional[Union[ivy.Array, ivy.NativeArray]] = None) \
        -> ivy.Array:
    """
    Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position. Array elements that roll beyond the first position are re-introduced at the last position.
    Parameters
    ----------
    x: array
        input array.
    shift: Union[int, Tuple[int, ...]]
        number of places by which the elements are shifted. If ``shift`` is a tuple, then ``axis`` must be a tuple of the same size, and each of the given axes must be shifted by the corresponding element in ``shift``. If ``shift`` is an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for all specified axes. If a shift is positive, then array elements must be shifted positively (toward larger indices) along the dimension of ``axis``. If a shift is negative, then array elements must be shifted negatively (toward smaller indices) along the dimension of ``axis``.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis (or axes) along which elements to shift. If ``axis`` is ``None``, the array must be flattened, shifted, and then restored to its original shape. Default: ``None``.
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
    return: array
        an output array having the same data type as ``x`` and whose elements, relative to ``x``, are shifted.
    """
    return _cur_framework(x).roll(x, shift, axis, out)


def squeeze(x: Union[ivy.Array, ivy.NativeArray],
            axis: Union[int, Tuple[int, ...]],
            out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
        -> ivy.Array:
    """
    Removes singleton dimensions (axes) from ``x``.
    Parameters
    ----------
    x: array
        input array.
    axis: Union[int, Tuple[int, ...]]
        axis (or axes) to squeeze. If a specified axis has a size greater than one, a ``ValueError`` must be raised.
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
    return: array
        an output array having the same data type and elements as ``x``.
    """
    return _cur_framework(x).squeeze(x, axis, out)


def flip(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
         out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
        -> ivy.Array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    Parameters
    ----------
    x:
        input array.
    axis:
        axis (or axes) along which to flip. If ``axis`` is ``None``, the function must flip all input array axes. If ``axis`` is negative, the function must count from the last dimension. If provided more than one axis, the function must flip only the specified axes. Default: ``None``.
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
    return:
        an output array having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
    """

    return _cur_framework(x).flip(x, axis, out)


def expand_dims(x: Union[ivy.Array, ivy.NativeArray],
                axis: Optional[Union[int, Tuple[int], List[int]]] = None,
                out: Optional[Union[ivy.Array, ivy.NativeArray]] = None) \
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
    return _cur_framework(x).expand_dims(x, axis, out)


def permute_dims(x: Union[ivy.Array, ivy.NativeArray],
                 axes: Tuple[int, ...],
                 out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
        -> ivy.Array:
    """
    Permutes the axes (dimensions) of an array x.

    Parameters
    ----------
    x:
        input array.
    axes:
        tuple containing a permutation of (0, 1, ..., N-1) where N is the number of axes (dimensions) of x.
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
    return:
        an array containing the axes permutation. The returned array must have the same data type as x.
    """
    return _cur_framework(x).permute_dims(x, axes, out)


def stack(arrays: Union[Tuple[ivy.Array], List[ivy.Array], Tuple[ivy.NativeArray], List[ivy.NativeArray]],
          axis: int = 0, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None) \
          -> ivy.Array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays:
        input arrays to join. Each array must have the same shape.
    axis:
        axis along which the arrays will be joined. Providing an ``axis`` specifies the index of the new axis in the dimensions of the result. For example, if ``axis`` is ``0``, the new axis will be the first dimension and the output array will have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the second dimension and the output array will have shape ``(A, N, B, C)``; and, if ``axis`` is ``-1``, the new axis will be the last dimension and the output array will have shape ``(A, B, C, N)``. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    --------
    return:
        an output array having rank ``N+1``, where ``N`` is the rank (number of dimensions) of ``x``. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.
        .. note::
           This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
    """

    return _cur_framework(arrays).stack(arrays, axis, out)


def reshape(x: Union[ivy.Array, ivy.NativeArray],
            shape: Tuple[int, ...],
            copy: Optional[bool] = None,
            out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
        -> ivy.Array:
    """
    Gives a new shape to an array without changing its data.

    :param x: Tensor to be reshaped.
    :type x: array
    :param newshape: The new shape should be compatible with the original shape. One shape dimension can be -1.
                        In this case, the value is inferred from the length of the array and remaining dimensions.
    :type newshape: int or sequence of ints
    :return: Reshaped array.
    """
    return _cur_framework(x).reshape(x, shape, copy, out)


def concat(xs: Union[Tuple[Union[ivy.Array, ivy.NativeArray], ...],List[Union[ivy.Array, ivy.NativeArray]]], axis: Optional[int] = 0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Casts an array to a specified type.

    :param xs: The input arrays must have the same shape, except in the dimension corresponding to axis
                        (the first, by default).
    :type xs: sequence of arrays
    :param axis: The axis along which the arrays will be joined. Default is -1.
    :type axis: int, optional
    :return: The concatenated array.
    """
    return _cur_framework(xs[0]).concat(xs, axis)


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
