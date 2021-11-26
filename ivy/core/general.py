"""
Collection of general Ivy functions.
"""

# global
import gc
import math
import einops
import numpy as np
from numbers import Number
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable

# local
import ivy
from ivy.core.device import dev_str
from ivy.framework_handler import current_framework as _cur_framework

FN_CACHE = dict()
INF = float('inf')
TIMEOUT = 15.0
TMP_DIR = '/tmp'


# noinspection PyShadowingNames
def array(object_in: Union[List, np.ndarray, ivy.Array, ivy.NativeArray], dtype_str: str = None,
          dev_str: str = None, f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Creates an array.

    :param object_in: An array_like object, which exposes the array interface,
            an object whose __array__ method returns an array, or any (nested) sequence.
    :type object_in: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype_str: data-type string, optional
    :param dev_str: device string on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array object satisfying the specified requirements, in the form of the selected framework.
    """
    return _cur_framework(object_in, f=f).array(object_in, dtype_str, dev_str)


def is_array(x: Any, exclusive: bool = False, f: ivy.Framework = None)\
        -> bool:
    """
    Determines whether the input x is an Ivy Array.

    :param x: The input to check
    :type x: any
    :param exclusive: Whether to check if the data type is exclusively an array, rather than a variable or traced array.
    :type exclusive: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, whether or not x is an array.
    """
    try:
        return _cur_framework(x, f=f).is_array(x, exclusive)
    except ValueError:
        return False


# noinspection PyShadowingNames
def copy_array(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Copy an array.

    :param x: The array to copy
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A copy of the input array.
    """
    return _cur_framework(x, f=f).copy_array(x)


def array_equal(x0: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> bool:
    """
    Determines whether two input arrays are equal across all elements.

    :param x0: The first input array to compare.
    :type x0: array
    :param x1: The second input array to compare.
    :type x1: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, whether or not the input arrays are equal across all elements.
    """
    return _cur_framework(x0, f=f).array_equal(x0, x1)


def arrays_equal(xs: List[Union[ivy.Array, ivy.NativeArray]])\
        -> bool:
    """
    Determines whether input arrays are equal across all elements.

    :param xs: Sequence of arrays to compare for equality
    :type xs: sequence of arrays
    :return: Boolean, whether or not all of the input arrays are equal across all elements.
    """
    x0 = xs[0]
    for x in xs[1:]:
        if not array_equal(x0, x):
            return False
    return True


def equal(*xs: Iterable[Any], equality_matrix: bool = False)\
        -> Union[bool, Union[ivy.Array, ivy.NativeArray]]:
    """
    Determines whether the inputs are all equal.

    :param xs: inputs to compare.
    :type xs: any
    :param equality_matrix: Whether to return a matrix of equalities comparing each input with every other.
                            Default is False.
    :type equality_matrix: bool, optional
    :return: Boolean, whether or not the inputs are equal, or matrix array of booleans if equality_matrix=True is set.
    """
    equality_fn = ivy.array_equal if ivy.is_array(xs[0]) else lambda a, b: a == b
    if equality_matrix:
        num_arrays = len(xs)
        mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if ivy.is_array(res):
                    # noinspection PyTypeChecker
                    res = ivy.to_scalar(res)
                # noinspection PyTypeChecker
                mat[i][j] = res
                # noinspection PyTypeChecker
                mat[j][i] = res
        return ivy.array(mat)
    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True


def to_numpy(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> np.ndarray:
    """
    Converts array into a numpy array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A numpy array.
    """
    return _cur_framework(x, f=f).to_numpy(x)


def to_scalar(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Number:
    """
    Converts an array with a single element into a scalar.

    :param x: Input array with a single element.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A scalar.
    """
    return _cur_framework(x, f=f).to_scalar(x)


def to_list(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> List:
    """
    Creates a (possibly nested) list from input array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list representation of the input array.
    """
    return _cur_framework(x, f=f).to_list(x)


def shape(x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False, f: ivy.Framework = None)\
        -> Iterable[int]:
    """
    Returns the shape of the array x.

    :param x: Input array to infer the shape of.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Shape of the array
    """
    return _cur_framework(x, f=f).shape(x, as_array)


def get_num_dims(x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False, f: ivy.Framework = None) -> int:
    """
    Returns the number of dimensions of the array x.

    :param x: Input array to infer the number of dimensions for.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Shape of the array
    """
    return _cur_framework(x, f=f).get_num_dims(x, as_array)


def minimum(x: Union[ivy.Array, ivy.NativeArray], y: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    :param x: Input array containing elements to minimum threshold.
    :type x: array
    :param y: Tensor containing minimum values, must be broadcastable to x.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but clipped to not exceed the y values.
    """
    return _cur_framework(x, f=f).minimum(x, y)


def maximum(x: Union[ivy.Array, ivy.NativeArray, Number], y: Union[ivy.Array, ivy.NativeArray, Number],
            f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    :param x: Input array containing elements to maximum threshold.
    :type x: array
    :param y: Tensor containing maximum values, must be broadcastable to x.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but clipped to not be lower than the y values.
    """
    return _cur_framework(x, f=f).maximum(x, y)


def clip(x: Union[ivy.Array, ivy.NativeArray], x_min: Union[Number, Union[ivy.Array, ivy.NativeArray]],
         x_max: Union[Number, Union[ivy.Array, ivy.NativeArray]], f: ivy.Framework = None)\
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
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the elements of x, but where values < x_min are replaced with x_min,
                and those > x_max with x_max.
    """
    return _cur_framework(x, f=f).clip(x, x_min, x_max)


def clip_vector_norm(x: Union[ivy.Array, ivy.NativeArray], max_norm: float, p: float = 2.0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Clips (limits) the vector p-norm of an array.

    :param x: Input array containing elements to clip.
    :type x: array
    :param max_norm: The maximum value of the array norm.
    :type max_norm: float
    :param p: The p-value for computing the p-norm. Default is 2.
    :type p: float, optional
    :return: An array with the vector norm downscaled to the max norm if needed.
    """
    norm = ivy.vector_norm(x, p, keepdims=True)
    ratio = ivy.stable_divide(max_norm, norm)
    if ratio < 1:
        return ratio * x
    return x


def clip_matrix_norm(x: Union[ivy.Array, ivy.NativeArray], max_norm: float, p: float = 2.0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Clips (limits) the matrix norm of an array.

    :param x: Input array containing elements to clip.
    :type x: array
    :param max_norm: The maximum value of the array norm.
    :type max_norm: float
    :param p: The p-value for computing the p-norm. Default is 2.
    :type p: float, optional
    :return: An array with the matrix norm downscaled to the max norm if needed.
    """
    norms = ivy.matrix_norm(x, p, keepdims=True)
    ratios = ivy.maximum(ivy.stable_divide(max_norm, norms), 1.)
    return ratios * x


# noinspection PyShadowingBuiltins
def round(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Rounds the values of an array to the nearest integer, element-wise.

    :param x: Input array containing elements to round.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements rounded to integers.
    """
    return _cur_framework(x, f=f).round(x)


def floormod(x: Union[ivy.Array, ivy.NativeArray], y: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns element-wise remainder of division.

    :param x: Input array to floormod.
    :type x: array
    :param y: Denominator input for floormod.
    :type y: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements floor modded.
    """
    return _cur_framework(x, f=f).floormod(x, y)


def floor(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns element-wise largest integer not greater than x.

    :param x: Input array to floor.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements floored to integers.
    """
    return _cur_framework(x, f=f).floor(x)


def ceil(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns element-wise smallest integer not less than x.

    :param x: Input array to ceil.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements ceiled to integers.
    """
    return _cur_framework(x, f=f).ceil(x)


# noinspection PyShadowingBuiltins
def abs(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the absolute value of each element in x.

    :param x: Input array containing elements to absolute value.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with all values now positive.
    """
    return _cur_framework(x, f=f).abs(x)


def argmax(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the index with the largest value across axes of an array.

    :param x: Input array containing elements to argmax.
    :type x: array
    :param axis: Axis to perform the argmax, default is 0.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor containing the indices of the maximum values across the specified axis.
    """
    return _cur_framework(x, f=f).argmax(x, axis)


def argmin(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the index with the smallest value across axes of an array.

    :param x: Input array containing elements to argmin.
    :type x: array
    :param axis: Axis to perform the argmin, default is 0.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor containing the indices of the minimum values across the specified axis.
    """
    return _cur_framework(x, f=f).argmin(x, axis)


def argsort(x: Union[ivy.Array, ivy.NativeArray], axis: int = -1, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the indices of a tensor that give its sorted order along an axis.

    :param x: Input array containing elements to argsort.
    :type x: array
    :param axis: Axis to perform the argsort, default is -1.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The indices that would sort each slice of the given values along the given axis.
    """
    return _cur_framework(x, f=f).argsort(x, axis)


# noinspection PyShadowingNames
def cast(x: Union[ivy.Array, ivy.NativeArray], dtype_str: str, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Casts an array to a specified type.

    :param x: Input array containing elements to cast.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
            If not given, then the type will be determined as the minimum type required to hold the objects in the
            sequence.
    :type dtype_str: data-type string
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with data type given by dtype_str.
    """
    return _cur_framework(x, f=f).cast(x, dtype_str)


# noinspection PyShadowingNames
def arange(stop: Number, start: Number = 0, step: Number = 1, dtype_str: str = None, dev_str: str = None,
           f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns evenly spaced values within a given interval, with the spacing being specified.

    Values are generated within the half-open interval [start, stop) (in other words, the interval including start but
    excluding stop). For integer arguments the function is equivalent to the Python built-in range function,
    but returns an array in the chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    :param stop: End of interval. The interval does not include this value, except in some cases where step is not an
                integer and floating point round-off affects the length of out.
    :type stop: number
    :param start: Start of interval. The interval includes this value. The default start value is 0.
    :type start: number, optional
    :param step: Spacing between values. For any output out, this is the distance between two adjacent values,
                    out[i+1] - out[i]. The default step size is 1. If step is specified as a position argument,
                    start must also be given.
    :type step: number, optional
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of evenly spaced values.

            For floating point arguments, the length of the result is ceil((stop - start)/step).
            Because of floating point overflow, this rule may result in the last element of out being greater than stop.
    """
    return _cur_framework(f=f).arange(stop, start, step, dtype_str, dev_str)


# noinspection PyShadowingNames
def linspace(start: Union[ivy.Array, ivy.NativeArray, Number], stop: Union[ivy.Array, ivy.NativeArray, Number],
             num: int, axis: int = None, dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Generates a certain number of evenly-spaced values in an interval along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in an interval.

    :param start: First entry in the range.
    :type start: array
    :param stop: Final entry in the range.
    :type stop: array
    :param num: Number of values to generate.
    :type num: int
    :param axis: Axis along which the operation is performed.
    :type axis: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of evenly-spaced values.
    """
    return _cur_framework(start, f=f).linspace(start, stop, num, axis, dev_str)


# noinspection PyShadowingNames
def logspace(start: Union[ivy.Array, ivy.NativeArray, Number], stop: Union[ivy.Array, ivy.NativeArray, Number],
             num: int, base: float = 10., axis: int = None, dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Generates a certain number of evenly-spaced values in log space, in an interval along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in an interval.

    :param start: First entry in the range.
    :type start: array
    :param stop: Final entry in the range.
    :type stop: array
    :param num: Number of values to generate.
    :type num: int
    :param base: The base of the log space. Default is 10.0
    :type base: float, optional
    :param axis: Axis along which the operation is performed.
    :type axis: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of evenly-spaced values.
    """
    return _cur_framework(start, f=f).logspace(start, stop, num, base, axis, dev_str)


def concatenate(xs: Iterable[Union[ivy.Array, ivy.NativeArray]], axis: int = -1, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Casts an array to a specified type.

    :param xs: The input arrays must have the same shape, except in the dimension corresponding to axis
                        (the first, by default).
    :type xs: sequence of arrays
    :param axis: The axis along which the arrays will be joined. Default is -1.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The concatenated array.
    """
    return _cur_framework(xs[0], f=f).concatenate(xs, axis)


def flip(x: Union[ivy.Array, ivy.NativeArray], axis: int = None, batch_shape: Iterable[int] = None,
         f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Reverses the ord of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    :param x: Input array.
    :type x: array
    :param axis: Axis or axes along which to flip over. The default, axis: int = None, will flip over all axes.
    :type axis: None or int or sequence of ints, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the entries of axis reversed.
    """
    return _cur_framework(x, f=f).flip(x, axis, batch_shape)


def stack(xs: Iterable[Union[ivy.Array, ivy.NativeArray]], axis: int = 0, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Joins a sequence of arrays along a new axis.
    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis: int = 0, it will be the first dimension and if axis: int = -1, it will be the last dimension.

    :param xs: Input arrays, each array must have the same shape.
    :type xs: sequence of arrays
    :param axis: The axis in the result array along which the input arrays are stacked.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The stacked array has one more dimension than the input arrays.
    """
    return _cur_framework(xs[0], f=f).stack(xs, axis)


def unstack(x: Union[ivy.Array, ivy.NativeArray], axis: int, keepdims: bool = False, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Unpacks the given dimension of a rank-R array into rank-(R-1) arrays.

    :param x: Input array to unstack.
    :type x: array
    :param axis: Axis for which to unpack the array.
    :type axis: int
    :param keepdims: Whether to keep dimension 1 in the unstack dimensions. Default is False.
    :type keepdims: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: List of arrays, unpacked along specified dimensions.
    """
    return _cur_framework(x, f=f).unstack(x, axis, keepdims)


def split(x: Union[ivy.Array, ivy.NativeArray], num_or_size_splits: Union[int, Iterable[int]] = None, axis: int = 0,
          with_remainder: bool = False, f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
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
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list of sub-arrays.
    """
    return _cur_framework(x, f=f).split(x, num_or_size_splits, axis, with_remainder)


def repeat(x: Union[ivy.Array, ivy.NativeArray], repeats: Union[int, Iterable[int]], axis: int = None,
           f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Repeat values along a given dimension

    :param x: Input array.
    :type x: array
    :param repeats: The number of repetitions for each element. repeats is broadcast to fit the shape of the given axis.
    :type repeats: int or sequence of ints.
    :param axis: The axis along which to repeat values.
                  By default, use the flattened input array, and return a flat output array.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The repeated output array.
    """
    return _cur_framework(x, f=f).repeat(x, repeats, axis)


def tile(x: Union[ivy.Array, ivy.NativeArray], reps: Iterable[int], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Constructs an array by repeating x the number of times given by reps.

    :param x: Input array.
    :type x: array
    :param reps: The number of repetitions of x along each axis.
    :type reps: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The tiled output array.
    """
    return _cur_framework(x, f=f).tile(x, reps)


def constant_pad(x: Union[ivy.Array, ivy.NativeArray], pad_width: Iterable[Tuple[int]], value: Number = 0,
                 f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Pads an array with a constant value.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :param value: The constant value to pad the array with.
    :type value: float or int, default zero
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _cur_framework(x, f=f).constant_pad(x, pad_width, value)


def zero_pad(x: Union[ivy.Array, ivy.NativeArray], pad_width: Iterable[Tuple[int]], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Pads an array with zeros.

    :param x: Input array to pad.
    :type x: array
    :param pad_width: Number of values padded to the edges of each axis.
                      Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    :type pad_width: sequence of tuples of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Padded array of rank equal to x with shape increased according to pad_width.
    """
    return _cur_framework(x, f=f).zero_pad(x, pad_width)


def fourier_encode(x: Union[ivy.Array, ivy.NativeArray], max_freq: float, num_bands: int = 4, linear: bool = False,
                   concat: bool = True, flatten: bool = False) -> Union[ivy.Array, ivy.NativeArray, Tuple]:
    """
    Pads an array with fourier encodings.

    :param x: Input array to encode.
    :type x: array
    :param max_freq: The maximum frequency of the encoding.
    :type max_freq: float
    :param num_bands: The number of frequency bands for the encoding. Default is 4.
    :type num_bands: int, optional
    :param linear: Whether to space the frequency bands linearly as opposed to geometrically. Default is False.
    :type linear: bool, optional
    :param concat: Whether to concatenate the position, sin and cos values, or return seperately. Default is True.
    :type concat: bool, optional
    :param flatten: Whether to flatten the position dimension into the batch dimension. Default is False.
    :type flatten: bool, optional
    :return: New array with the final dimension expanded, and the encodings stored in this channel.
    """
    x_in = x
    dim = x.shape[-1]
    x = ivy.expand_dims(x, -1)
    orig_x = x
    if linear:
        scales = ivy.linspace(1., max_freq / 2, num_bands, dev_str=dev_str(x))
    else:
        scales = ivy.logspace(0., math.log(max_freq / 2) / math.log(10), num_bands, base=10, dev_str=dev_str(x))
    scales = ivy.cast(scales, ivy.dtype_str(x))
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * math.pi
    sin_x = ivy.sin(x)
    cos_x = ivy.cos(x)
    if flatten:
        orig_x = x_in
        sin_x = ivy.reshape(sin_x, [-1, num_bands*dim])
        cos_x = ivy.reshape(cos_x, [-1, num_bands*dim])
    if concat:
        return ivy.concatenate([orig_x, sin_x, cos_x], -1)
    return sin_x, cos_x


def swapaxes(x: Union[ivy.Array, ivy.NativeArray], axis0: int, axis1: int, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Interchange two axes of an array.

    :param x: Input array.
    :type x: array
    :param axis0: First axis to be swapped.
    :type axis0: int
    :param axis1: Second axis to be swapped.
    :type axis1: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: x with its axes permuted.
    """
    return _cur_framework(x, f=f).swapaxes(x, axis0, axis1)


def transpose(x: Union[ivy.Array, ivy.NativeArray], axes: Iterable[int] = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Permutes the dimensions of an array.

    :param x: Input array.
    :type x: array
    :param axes: By default, reverse the dimensions, otherwise permute the axes according to the values given.
    :type axes: sequence of ints of length N
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: x with its axes permuted.
    """
    return _cur_framework(x, f=f).transpose(x, axes)


def expand_dims(x: Union[ivy.Array, ivy.NativeArray], axis: int, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Expands the shape of an array.
    Inserts a new axis that will appear at the axis position in the expanded array shape.

    :param x: Input array.
    :type x: array
    :param axis: Position in the expanded axes where the new axis is placed.
    :type axis: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: array with the number of dimensions increased by onearray
    """
    return _cur_framework(x, f=f).expand_dims(x, axis)


def where(condition: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray],
          x2: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns elements chosen from x or y depending on condition.

    :param condition: Where True, yield x1, otherwise yield x2.
    :type condition: bool array
    :param x1: values from which to choose when condition is True.
    :type x1: array
    :param x2: values from which to choose when condition is False.
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with elements from x1 where condition is True, and elements from x2 elsewhere.
    """
    return _cur_framework(x1, f=f).where(condition, x1, x2)


def indices_where(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns indices or true elements in an input boolean array.

    :param x: Boolean array, for which indices are desired.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Indices for where the boolean array is True.
    """
    return _cur_framework(x, f=f).indices_where(x)


def isnan(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns boolean map at locations where the input is not a number (nan).

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean values for where the values of the array are nan.
    """
    return _cur_framework(x, f=f).isnan(x)


def value_is_nan(x: Union[ivy.Array, ivy.NativeArray, Number], include_infs: bool = True)\
        -> bool:
    """
    Determine whether the single valued array or scalar is of nan type

    :param x: The input to check Input array.
    :type x: array
    :param include_infs: Whether to include infs and -infs in the check. Default is True.
    :type include_infs: bool, optional
    :return Boolean as to whether the input value is a nan or not.
    """
    x_scalar = ivy.to_scalar(x) if ivy.is_array(x) else x
    if not x_scalar == x_scalar:
        return True
    if include_infs and x_scalar == INF or x_scalar == -INF:
        return True
    return False


def has_nans(x: Union[ivy.Array, ivy.NativeArray], include_infs: bool = True)\
        -> bool:
    """
    Determine whether the array contains any nans, as well as infs or -infs if specified.

    :param x: Input array.
    :type x: array
    :param include_infs: Whether to include infs and -infs in the check. Default is True.
    :type include_infs: bool, optional
    :return: Boolean as to whether the array contains nans.
    """
    return value_is_nan(ivy.reduce_sum(x), include_infs)


def reshape(x: Union[ivy.Array, ivy.NativeArray], newshape: Union[int, Iterable[int]], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gives a new shape to an array without changing its data.

    :param x: Tensor to be reshaped.
    :type x: array
    :param newshape: The new shape should be compatible with the original shape. One shape dimension can be -1.
                        In this case, the value is inferred from the length of the array and remaining dimensions.
    :type newshape: int or sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Reshaped array.
    """
    return _cur_framework(x, f=f).reshape(x, newshape)


def broadcast_to(x: Union[ivy.Array, ivy.NativeArray], newshape: Iterable[int], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Broadcast the input tensor to newshape, adding dimensions of size 1 where the dimensions do not align.

    :param x: Tensor to be broadcast to new shape.
    :type x: array
    :param newshape: The new shape the tensor should be broadcast to.
    :type newshape: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Newly broadcast array.
    """
    return _cur_framework(x, f=f).broadcast_to(x, newshape)


def squeeze(x: Union[ivy.Array, ivy.NativeArray], axis: int = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Removes a single-dimensional entry from the shape of an array.

    :param x: Input data.
    :type x: array
    :param axis: Index for one of the single-dimensional entries in the shape.
                 If an axis is selected with shape entry greater than one, an error is raised.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array, but with all (axis=None) or one (axis is int) of the dimensions of length 1 removed.
    """
    return _cur_framework(x, f=f).squeeze(x, axis)


# noinspection PyShadowingNames
def zeros(shape: Iterable[int], dtype_str: str = 'float32', dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Return a new array of given shape and type, filled with zeros.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the given shape and dtype_str.
    """
    return _cur_framework(f=f).zeros(shape, dtype_str, dev_str)


# noinspection PyShadowingNames
def zeros_like(x: Union[ivy.Array, ivy.NativeArray], dtype_str: str = None, dev_str: str = None,
               f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns an array of zeros with the same shape and type as x, unless dtype_str provided which overrides.

    :param x: The shape and data-type of x define these same attributes of the returned array.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype_str provided which overrides.
    """
    return _cur_framework(x, f=f).zeros_like(x, dtype_str, dev_str)


# noinspection PyShadowingNames
def ones(shape: Iterable[int], dtype_str: str = 'float32', dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of ones with the given shape and dtype_str.
    """
    return _cur_framework(f=f).ones(shape, dtype_str, dev_str)


# noinspection PyShadowingNames
def ones_like(x: Union[ivy.Array, ivy.NativeArray], dtype_str: str = None, dev_str: str = None,
              f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns an array of ones with the same shape and type as x, unless dtype_str provided which overrides.

    :param x: The shape and data-type of a define these same attributes of the returned array.
    :type x: array
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype_str: data-type string, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype_str provided which overrides.
    """
    return _cur_framework(x, f=f).ones_like(x, dtype_str, dev_str)


# noinspection PyShadowingNames
def one_hot(indices: Union[ivy.Array, ivy.NativeArray], depth: int, dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a one-hot array
    :param indices: Indices for where the ones should be scattered *[batch_shape, dim]*
    :type indices: array
    :param depth: Scalar defining the depth of the one-hot dimension.
    :type depth: int
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype provided which overrides.
    """
    return _cur_framework(indices, f=f).one_hot(indices, depth, dev_str)


def cross(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the cross product of two (arrays of) vectors in R^3.
    The cross product of x1 and x2 in R^3 is a vector perpendicular to both x1 and x2.
    If x1 and x2 are arrays of vectors, the vectors are defined by the last axis of x1 and x2 by default which must have
    dimension 3.

    :param x1: Components of the first vector(s).
    :type x1: array
    :param x2: Components of the second vector(s).
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Vector cross product(s).
    """
    return _cur_framework(x1, f=f).cross(x1, x2)


def matmul(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Computes the matrix product of two arrays x1 and x2.

    :param x1: Input array 1.
    :type x1: array
    :param x2: Input array 2.
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The matrix product of the input arrays.
    """
    return _cur_framework(x1, f=f).matmul(x1, x2)


def cumsum(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the cumulative sum of the elements along a given axis.

    :param x: Input array.
    :type x: array
    :param axis: Axis along which the cumulative sum is computed. By default 0.
    :type axis: int
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Input array with cumulatively summed elements along axis.
    """
    return _cur_framework(x, f=f).cumsum(x, axis)


def cumprod(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0, exclusive: bool = False, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the cumulative product of the elements along a given axis.

    :param x: Input array.
    :type x: array
    :param axis: Axis along which the cumulative product is computed. By default 0.
    :type axis: int
    :param exclusive: Whether to perform the cumprod exclusively. Defaults is False.
    :type exclusive: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Input array with cumulatively multiplied elements along axis.
    """
    return _cur_framework(x, f=f).cumprod(x, axis, exclusive)


# noinspection PyShadowingNames
def identity(n: int, dtype_str: str = 'float32', batch_shape: Iterable[int] = None, dev_str: str = None,
             f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the identity array.
    The identity array is a square array with ones on the main diagonal.

    :param n: Number of rows (and columns) in n x n output.
    :type n: int
    :param dtype_str: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                      Default is 'float32'.
    :type dtype_str: data-type string, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: n x n array of type dtype_str, with its main diagonal set to one, and all other elements 0.
    """
    return _cur_framework(f=f).identity(n, dtype_str, batch_shape, dev_str)


def meshgrid(*xs: Iterable[Union[ivy.Array, ivy.NativeArray]], indexing: str = 'ij', f: ivy.Framework = None)\
        -> Iterable[Union[ivy.Array, ivy.NativeArray]]:
    """
    Broadcasts parameters for evaluation on an N-D grid.

    :param xs: input arrays
    :type xs: sequence of arrays
    :param indexing: The indexing method, either 'xy' or 'ij'. Default is 'ij'.
    :type indexing: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: list of N-D coordinate arrays for evaluating expressions on an N-D grid
    """
    return _cur_framework(f=f).meshgrid(*xs, indexing=indexing)


# noinspection PyShadowingNames
def scatter_flat(indices: Union[ivy.Array, ivy.NativeArray], updates: Union[ivy.Array, ivy.NativeArray], size: int,
                 reduction: str = 'sum', dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Scatter flat updates into a new flat array according to flat indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param size: The size of the result.
    :type size: int
    :param reduction: The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    :type reduction: str
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _cur_framework(indices, f=f).scatter_flat(indices, updates, size, reduction, dev_str)


# noinspection PyShadowingNames
def scatter_nd(indices: Union[ivy.Array, ivy.NativeArray], updates: Union[ivy.Array, ivy.NativeArray],
               shape: Iterable[int], reduction: str = 'sum', dev_str: str = None, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Scatter updates into a new array according to indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param shape: The shape of the result.
    :type shape: sequence of ints
    :param reduction: The reduction method for the scatter, one of 'sum', 'min' or 'max'
    :type reduction: str
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _cur_framework(indices, f=f).scatter_nd(indices, updates, shape, reduction, dev_str)


# noinspection PyShadowingNames
def gather(params: Union[ivy.Array, ivy.NativeArray], indices: Union[ivy.Array, ivy.NativeArray], axis: int = -1,
           dev_str: str = None, f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gather slices from params at axis according to indices.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param axis: The axis from which to gather from. Default is -1.
    :type axis: int, optional
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array with the values gathered at the specified indices along the specified axis.
    """
    return _cur_framework(params, f=f).gather(params, indices, axis, dev_str)


# noinspection PyShadowingNames
def gather_nd(params: Union[ivy.Array, ivy.NativeArray], indices: Union[ivy.Array, ivy.NativeArray],
              dev_str: str = None, f: ivy.Framework = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gather slices from params into a array with shape specified by indices.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param dev_str: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: New array of given shape, with the values gathered at the indices.
    """
    return _cur_framework(params, f=f).gather_nd(params, indices, dev_str)


def linear_resample(x: Union[ivy.Array, ivy.NativeArray], num_samples: int, axis: int = -1, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Performs linear re-sampling on input image.

    :param x: Input array
    :type x: array
    :param num_samples: The number of interpolated samples to take.
    :type num_samples: int
    :param axis: The axis along which to perform the resample. Default is last dimension.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The array after the linear resampling.
    """
    return _cur_framework(x, f=f).linear_resample(x, num_samples, axis)


def exists(x: Any)\
        -> bool:
    """
    Simple check as to whether the input is None or not.

    :param x: Input to check.
    :type x: any
    :return: True if x is not None, else False.
    """
    return x is not None


def default(x: Any, default_val: Any, catch_exceptions: bool = False, rev: bool = False, with_callable: bool = False)\
        -> Any:
    """
    Returns x provided it exists (is not None), else returns default value.

    :param x: Input which may or may not exist (be None).
    :type x: value if catch_exceptions=False else callable
    :param default_val: The default value.
    :type default_val: any
    :param catch_exceptions: Whether to catch exceptions from callable x. Default is False.
    :type catch_exceptions: bool, optional
    :param rev: Whether to reverse the input x and default_val. Default is False.
    :type rev: bool, optional
    :param with_callable: Whether either of the arguments might be callable functions. Default is False.
    :type with_callable: bool, optional
    :return: x if x exists (is not None), else default.
    """
    with_callable = catch_exceptions or with_callable
    if rev:
        tmp = x
        x = default_val
        default_val = tmp
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        # noinspection PyBroadException
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return x if exists(x) else default_val() if default_callable else default_val


def try_else_none(fn):
    """
    Try and return the function, otherwise return None if an exception was raised during function execution.

    :param fn: Function to try and call and return.
    :type fn: callable
    """
    return default(fn, None, True)


def dtype(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> ivy.Dtype:
    """
    Get the data type for input array x.

    :param x: Tensor for which to get the data type.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Data type of the array
    """
    return _cur_framework(x, f=f).dtype(x)


def dtype_to_str(dtype_in: ivy.Dtype, f: ivy.Framework = None)\
        -> str:
    """
    Convert native data type to string representation.

    :param dtype_in: The data type to convert to string.
    :type dtype_in: data type
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _cur_framework(None, f=f).dtype_to_str(dtype_in)


def dtype_str(x: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> str:
    """
    Get the data type string for input array x.

    :param x: Tensor for which to get the data type string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _cur_framework(None, f=f).dtype_str(x)


def cache_fn(func: Callable)\
        -> Callable:
    """
    Wrap a function, such that when cache=True is passed as an argument, a previously cached output is returned.

    :param func: The function to wrap, whose output should be cached for later.
    :type func: callable
    :return: The newly cache wrapped function.
    """

    global FN_CACHE
    if func not in FN_CACHE:
        FN_CACHE[func] = dict()

    def cached_fn(*args, **kwargs):
        key = ''.join([str(i) + ', ' for i in args] + [' kw, '] + [str(i) + ', ' for i in sorted(kwargs.items())])
        cache = FN_CACHE[func]
        if key in cache:
            return cache[key]
        ret = func(*args, **kwargs)
        cache[key] = ret
        return ret

    return cached_fn


def current_framework_str(f: ivy.Framework = None)\
        -> Union[str, None]:
    """
    Return the string of the current globally set framework. Returns None if no framework is set.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The framework string.
    """
    fw = _cur_framework(f=f)
    if fw is None:
        return None
    return fw.current_framework_str()


def einops_rearrange(x: Union[ivy.Array, ivy.NativeArray], pattern: str, **axes_lengths: Dict[str, int])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Perform einops rearrange operation on input array x.

    :param x: Input array to be re-arranged.
    :type x: array
    :param pattern: Rearrangement pattern.
    :type pattern: str
    :param axes_lengths: Any additional specifications for dimensions.
    :type axes_lengths: keyword parameter args
    :return: New array with einops.rearrange having been applied.
    """
    return einops.rearrange(x, pattern, **axes_lengths)


def einops_reduce(x: Union[ivy.Array, ivy.NativeArray], pattern: str, reduction: Union[str, Callable],
                  **axes_lengths: Dict[str, int]) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Perform einops reduce operation on input array x.

    :param x: Input array to be reduced.
    :type x: array
    :param pattern: Reduction pattern.
    :type pattern: str
    :param reduction: One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or callable.
    :type reduction: str or callable
    :param axes_lengths: Any additional specifications for dimensions.
    :type axes_lengths: keyword parameter args
    :return: New array with einops.reduce having been applied.
    """
    return einops.reduce(x, pattern, reduction, **axes_lengths)


def einops_repeat(x: Union[ivy.Array, ivy.NativeArray], pattern: str, **axes_lengths: Dict[str, int])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Perform einops repeat operation on input array x.

    :param x: Input array to be repeated.
    :type x: array
    :param pattern: Rearrangement pattern.
    :type pattern: str
    :param axes_lengths: Any additional specifications for dimensions.
    :type axes_lengths: keyword parameter args
    :return: New array with einops.repeat having been applied.
    """
    return einops.repeat(x, pattern, **axes_lengths)


def get_min_denominator()\
        -> float:
    """
    Get the global minimum denominator used by ivy for numerically stable division.
    """
    # noinspection PyProtectedMember
    return ivy._MIN_DENOMINATOR


def set_min_denominator(val: float)\
        -> None:
    """
    Set the global minimum denominator used by ivy for numerically stable division.

    :param val: The new value to set the minimum denominator to.
    :type val: float
    """
    ivy._MIN_DENOMINATOR = val


def stable_divide(numerator: Any, denominator: Any, min_denominator: float = None) -> Any:
    """
    Divide the numerator by the denominator, with min denominator added to the denominator for numerical stability.

    :param numerator: The numerator of the division.
    :type numerator: any valid numerator, including containers
    :param denominator: The denominator of the division.
    :type denominator: any valid denominator, including containers
    :param min_denominator: The minimum denominator to use, use global ivy._MIN_DENOMINATOR by default.
    :type min_denominator: float, optional
    :return: The new item following the numerically stable division.
    """
    # noinspection PyProtectedMember
    return numerator / (denominator + default(min_denominator, ivy._MIN_DENOMINATOR))


def get_min_base()\
        -> float:
    """
    Get the global minimum base used by ivy for numerically stable power raising.
    """
    # noinspection PyProtectedMember
    return ivy._MIN_BASE


def set_min_base(val: float)\
        -> None:
    """
    Set the global minimum base used by ivy for numerically stable power raising.

    :param val: The new value to set the minimum base to.
    :type val: float
    """
    ivy._MIN_BASE = val


def stable_pow(base: Any, exponent: Any, min_base: float = None)\
        -> Any:
    """
    Raise the base by the power, with MIN_BASE added to the base when exponent > 1 for numerical stability.

    :param base: The numerator of the division.
    :type base: any valid numerator, including containers
    :param exponent: The denominator of the division.
    :type exponent: any valid denominator, including containers
    :param min_base: The minimum base to use, use global ivy._MIN_BASE by default.
    :type min_base: float, optional
    :return: The new item following the numerically stable division.
    """
    # noinspection PyProtectedMember
    return (base + default(min_base, ivy._MIN_BASE)) ** exponent


def multiprocessing(context: str = None, f: ivy.Framework = None):
    """
    Return framewrk-specific multi-processing module

    :param context: The context of the multiprocessing, either fork, forkserver or spawn. Default is None.
    :type context: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Multiprocessing module
    """
    return _cur_framework(f=f).multiprocessing(context)


def set_queue_timeout(timeout):
    """
    Set the global queue timeout values (in seconds). Default value without this function being called is 10 seconds.

    :param timeout: The timeout to set in seconds.
    :type timeout: float, optional
    """
    global TIMEOUT
    TIMEOUT = timeout


def queue_timeout():
    """
    Get the global queue timeout values (in seconds). Default value without this function being called is 10 seconds.
    """
    global TIMEOUT
    return TIMEOUT


def tmp_dir():
    """
    Return the directory for saving temporary files.
    """
    return TMP_DIR


def set_tmp_dir(tmp_dr):
    """
    Set the directory for saving temporary files.
    """
    global TMP_DIR
    TMP_DIR = tmp_dr


def get_all_arrays_in_memory():
    """
    Gets all arrays which are currently alive.
    """
    all_arrays = list()
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if ivy.is_array(obj):
                all_arrays.append(obj)
        except Exception:
            pass
    return all_arrays


def num_arrays_in_memory():
    """
    Returns the number of arrays which are currently alive.
    """
    return len(get_all_arrays_in_memory())


def print_all_arrays_in_memory():
    """
    Prints all arrays which are currently alive.
    """
    for arr in get_all_arrays_in_memory():
        print(type(arr), arr.shape)


def container_types(f: ivy.Framework = None):
    """
    Return all framework-specific types which should be hierarchically parsed in an ivy.Container. Such types must adopt
    a key-value structure, and exposes public methods .keys(), .values() and items().
    """
    # noinspection PyBroadException
    try:
        return _cur_framework(f=f).container_types()
    except ValueError:
        return []
