"""
Collection of general Ivy functions.
"""

# global
import math
import einops
import nvidia_smi
import numpy as np
from psutil import virtual_memory

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

FN_CACHE = dict()


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev_str=None, f=None):
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


def is_array(x, f=None):
    """
    Determines whether the input x is an Ivy Array.

    :param x: The input to check
    :type x: any
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, whether or not x is an array.
    """
    try:
        return _cur_framework(x, f=f).is_array(x)
    except ValueError:
        return False


def array_equal(x0, x1, f=None):
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


def equal(*xs, equality_matrix=False):
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
                    res = ivy.to_scalar(res)
                mat[i][j] = res
                mat[j][i] = res
        return ivy.array(mat)
    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True


def to_numpy(x, f=None):
    """
    Converts array into a numpy array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A numpy array.
    """
    return _cur_framework(x, f=f).to_numpy(x)


def to_scalar(x, f=None):
    """
    Converts an array with a single element into a scalar.

    :param x: Input array with a single element.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A scalar.
    """
    return _cur_framework(x, f=f).to_scalar(x)


def to_list(x, f=None):
    """
    Creates a (possibly nested) list from input array.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list representation of the input array.
    """
    return _cur_framework(x, f=f).to_list(x)


def shape(x, as_array=False, f=None):
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


def get_num_dims(x, as_array=False, f=None):
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


def minimum(x, y, f=None):
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


def maximum(x, y, f=None):
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


def clip(x, x_min, x_max, f=None):
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


def clip_vector_norm(x, max_norm: float, p: float = 2.0):
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
        return x * ratio
    return x


def clip_matrix_norm(x, max_norm: float, p: float = 2.0):
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
    return x * ratios


# noinspection PyShadowingBuiltins
def round(x, f=None):
    """
    Rounds the values of an array to the nearest integer, element-wise.

    :param x: Input array containing elements to round.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements rounded to integers.
    """
    return _cur_framework(x, f=f).round(x)


def floormod(x, y, f=None):
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


def floor(x, f=None):
    """
    Returns element-wise largest integer not greater than x.

    :param x: Input array to floor.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array of the same shape and type as x, with the elements floored to integers.
    """
    return _cur_framework(x, f=f).floor(x)


def ceil(x, f=None):
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
def abs(x, f=None):
    """
    Returns the absolute value of each element in x.

    :param x: Input array containing elements to absolute value.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with all values now positive.
    """
    return _cur_framework(x, f=f).abs(x)


def argmax(x, axis=0, f=None):
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


def argmin(x, axis=0, f=None):
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


def argsort(x, axis=-1, f=None):
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
def cast(x, dtype_str, f=None):
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
def arange(stop, start=0, step=1, dtype_str=None, dev_str=None, f=None):
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
def linspace(start, stop, num, axis=None, dev_str=None, f=None):
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
def logspace(start, stop, num, base=10., axis=None, dev_str=None, f=None):
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


def concatenate(xs, axis=-1, f=None):
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


def flip(x, axis=None, batch_shape=None, f=None):
    """
    Reverses the ord of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    :param x: Input array.
    :type x: array
    :param axis: Axis or axes along which to flip over. The default, axis=None, will flip over all axes.
    :type axis: None or int or sequence of ints, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: An array with the entries of axis reversed.
    """
    return _cur_framework(x, f=f).flip(x, axis, batch_shape)


def stack(xs, axis=0, f=None):
    """
    Joins a sequence of arrays along a new axis.
    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis=0, it will be the first dimension and if axis=-1, it will be the last dimension.

    :param xs: Input arrays, each array must have the same shape.
    :type xs: sequence of arrays
    :param axis: The axis in the result array along which the input arrays are stacked.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The stacked array has one more dimension than the input arrays.
    """
    return _cur_framework(xs[0], f=f).stack(xs, axis)


def unstack(x, axis, keepdims=False, f=None):
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


def split(x, num_or_size_splits=None, axis=0, with_remainder=False, f=None):
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
    :param with_remainder: If the tensor does not split evenly, then store the last remainder entry. Defaul is False.
    :type with_remainder: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A list of sub-arrays.
    """
    return _cur_framework(x, f=f).split(x, num_or_size_splits, axis, with_remainder)


def repeat(x, repeats, axis=None, f=None):
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


def tile(x, reps, f=None):
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


def constant_pad(x, pad_width, value=0, f=None):
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


def zero_pad(x, pad_width, f=None):
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


def fourier_encode(x, max_freq, num_bands=4, base=2.):
    """
    Pads an array with fourier encodings.

    :param x: Input array to encode.
    :type x: array
    :param max_freq: The maximum frequency of the encoding.
    :type max_freq: float
    :param num_bands: The number of frequency bands for the encoding. Default is 4.
    :type num_bands: int, optional
    :param base: The base of the encoding.
    :type base: float, optional
    :return: New array with the final dimension expanded, and the encodings stored in this channel.
    """
    x = ivy.expand_dims(x, -1)
    orig_x = x
    scales = ivy.cast(ivy.logspace(0., math.log(max_freq / 2) / math.log(base), num_bands,
                                   base=base, dev_str=dev_str(x)), ivy.dtype_str(x))
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    x = x * scales * math.pi
    return ivy.concatenate((orig_x, ivy.sin(x), ivy.cos(x)), -1)


def swapaxes(x, axis0, axis1, f=None):
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


def transpose(x, axes=None, f=None):
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


def expand_dims(x, axis, f=None):
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


def where(condition, x1, x2, f=None):
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


def indices_where(x, f=None):
    """
    Returns indices or true elements in an input boolean array.

    :param x: Boolean array, for which indices are desired.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Indices for where the boolean array is True.
    """
    return _cur_framework(x, f=f).indices_where(x)


def isnan(x, f=None):
    """
    Returns boolean map at locations where the input is not a number (nan).

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean values for where the values of the array are nan.
    """
    return _cur_framework(x, f=f).isnan(x)


def reshape(x, newshape, f=None):
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


def broadcast_to(x, newshape, f=None):
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


def squeeze(x, axis=None, f=None):
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
def zeros(shape, dtype_str='float32', dev_str=None, f=None):
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
def zeros_like(x, dtype_str=None, dev_str=None, f=None):
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
def ones(shape, dtype_str='float32', dev_str=None, f=None):
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
def ones_like(x, dtype_str=None, dev_str=None, f=None):
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
def one_hot(indices, depth, dev_str=None, f=None):
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


def cross(x1, x2, f=None):
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


def matmul(x1, x2, f=None):
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


def cumsum(x, axis=0, f=None):
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


def cumprod(x, axis=0, exclusive=False, f=None):
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
def identity(n, dtype_str='float32', batch_shape=None, dev_str=None, f=None):
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


def meshgrid(*xs, indexing='ij', f=None):
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
def scatter_flat(indices, updates, size, reduction='sum', dev_str=None, f=None):
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
def scatter_nd(indices, updates, shape, reduction='sum', dev_str=None, f=None):
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
def gather(params, indices, axis=-1, dev_str=None, f=None):
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
def gather_nd(params, indices, dev_str=None, f=None):
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


def linear_resample(x, num_samples, axis=-1, f=None):
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


def exists(x):
    """
    Simple check as to whether the input is None or not.

    :param x: Input to check.
    :type x: any
    :return: True if x is not None, else False.
    """
    return x is not None


def default(x, default_val, catch_exceptions=False):
    """
    Returns x provided it exists (is not None), else returns default value.

    :param x: Input which may or may not exist (be None).
    :type x: value if catch_exceptions=False else callable
    :param default_val: The default value.
    :type default_val: any
    :param catch_exceptions: Whether to catch exceptions from callable x. Default is False.
    :type catch_exceptions: bool, optional
    :return: x if x exists (is not None), else default.
    """
    if catch_exceptions:
        assert callable(x)
        # noinspection PyBroadException
        try:
            x = x()
        except Exception:
            return default_val
    return x if exists(x) else default_val


def dev(x, f=None):
    """
    Get the native device handle for input array x.

    :param x: Tensor for which to get the device handle.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device handle for the array, in native framework format.
    """
    return _cur_framework(x, f=f).dev(x)


# noinspection PyShadowingNames
def to_dev(x, dev_str=None, f=None):
    """
    Move the input array x to the desired device, specified by device string.

    :param x: Array to move onto the device.
    :type x: array
    :param dev_str: device to move the array to 'cuda:0', 'cuda:1', 'cpu' etc. Keep same device if None.
    :type dev_str: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The array x, but now placed on the target device.
    """
    return _cur_framework(x, f=f).to_dev(x, dev_str)


def dev_to_str(dev_in, f=None):
    """
    Convert native data type to string representation.

    :param dev_in: The device handle to convert to string.
    :type dev_in: device handle
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'cuda:0'.
    """
    return _cur_framework(None, f=f).dev_to_str(dev_in)


# noinspection PyShadowingNames
def str_to_dev(dev_str, f=None):
    """
    Convert device string representation to native device type.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Native device handle.
    """
    return _cur_framework(None, f=f).str_to_dev(dev_str)


def dev_str(x, f=None):
    """
    Get the device string for input array x.

    :param x: Tensor for which to get the device string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string for the array, e.g. 'cuda:0', 'cuda:1', 'cpu' etc..
    """
    return _cur_framework(x, f=f).dev_str(x)


# noinspection PyShadowingNames
def memory_on_dev(dev_str):
    """
    Get the total amount of memory for a given device string. In case of CPU, the total RAM is returned.

    :param dev_str: The device string to conver to native device handle.
    :type dev_str: str
    :return: The total memory on the device in GB.
    """
    if 'gpu' in dev_str or 'cuda' in dev_str:
        gpu_idx = int(dev_str.split(':')[-1])
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.total/1e9
    elif 'cpu' in dev_str:
        return virtual_memory().total/1e9
    else:
        raise Exception('Invalid device string input, must be on the form "gpu:idx" or "cpu:idx",'
                        'but found {}'.format(dev_str))


def gpu_is_available(f=None):
    """
    Determine whether a GPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a gpu is available.
    """
    return _cur_framework(f=f).gpu_is_available()


def num_gpus(f=None):
    """
    Determine the number of available GPUs, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Number of available GPUs.
    """
    return _cur_framework(f=f).num_gpus()


def tpu_is_available(f=None):
    """
    Determine whether a TPU is available to use, with the backend framework.

    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Boolean, as to whether a tpu is available.
    """
    return _cur_framework(f=f).tpu_is_available()


def dtype(x, f=None):
    """
    Get the data type for input array x.

    :param x: Tensor for which to get the data type.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Data type of the array
    """
    return _cur_framework(x, f=f).dtype(x)


def dtype_to_str(dtype_in, f=None):
    """
    Convert native data type to string representation.

    :param dtype_in: The data type to convert to string.
    :type dtype_in: data type
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _cur_framework(None, f=f).dtype_to_str(dtype_in)


def dtype_str(x, f=None):
    """
    Get the data type string for input array x.

    :param x: Tensor for which to get the data type string.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Device string e.g. 'float32'.
    """
    return _cur_framework(None, f=f).dtype_str(x)


def compile_fn(func, dynamic=True, example_inputs=None, f=None):
    """
    Provide a function which should be compiled, for faster inference.
    The handle to the newly compiled function is returned.

    :param func: Function to be compiled.
    :type func: callable
    :param dynamic: Whether to compile all conditional branches, regardless of inputs during first invocation.
    :type dynamic: bool, default True
    :param example_inputs: Example of inputs to the function to be compiled.
                            Required for torch in non-dynamic mode, unused by other frameworks.
    :type example_inputs: single input of tuple of inputs.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The handle to the newly compiled function.
    """
    return _cur_framework(example_inputs, f=f).compile_fn(func, dynamic, example_inputs)


def split_func_call(func, inputs, chunk_size, input_axes=0, output_axes=None, mean=False):
    """
    Call a function by splitting its inputs along a given axis, and calling the function in chunks, rather than feeding
    the entire input array at once. This can be useful to reduce memory usage of the device the arrays are on.

    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays
    :param chunk_size: The size of each of the chunks to be fed into the function.
    :type chunk_size: int
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :param mean: Whether to compute a weighted mean based on the return from each chunk. Default is False.
    :type mean: bool, optional
    :return: The return from the function, following input splitting and re-concattenation.
    """
    if isinstance(input_axes, int):
        input_axes = [input_axes]*len(inputs)
    dim_size = inputs[0].shape[input_axes[0]]
    num_chunks = dim_size / chunk_size
    num_chunks_floored = math.floor(dim_size / chunk_size)
    chunk_sizes = [chunk_size]*num_chunks_floored
    if num_chunks != num_chunks_floored:
        chunk_sizes.append(dim_size - chunk_size * num_chunks_floored)
    inputs_split = [ivy.split(inp, chunk_sizes, input_axes[i], True) if ivy.is_array(inp)
                    else inp.split(chunk_sizes, input_axes[i], True) for i, inp in enumerate(inputs)]
    rets = [func(*i) for i in zip(*inputs_split)]
    num_outputs = len(rets[0])
    if output_axes is None:
        output_axes = [input_axes[0]] * num_outputs
    elif isinstance(output_axes, int):
        output_axes = [output_axes] * num_outputs
    if mean:
        rets = [[(r.expand_dims(output_axis) if isinstance(r, ivy.Container) else ivy.expand_dims(r, output_axis)) * cs
                 for output_axis, r in zip(output_axes, ret)] for ret, cs in zip(rets, chunk_sizes)]
    concatted = [ivy.concatenate([r[i] for r in rets], output_axes[i]) if ivy.is_array(rets[0][i])
                 else ivy.Container.concat([r[i] for r in rets], output_axes[i])
                 for i in range(num_outputs)]
    if mean:
        return [(item.reduce_sum(output_axis) if isinstance(item, ivy.Container)
                 else ivy.reduce_sum(item, output_axis))/sum(chunk_sizes)
                for item, output_axis in zip(concatted, output_axes)]
    return concatted


def split_func_call_across_gpus(func, inputs, dev_strs, input_axes=0, output_axes=None, concat_output=False):
    """
    Call a function by splitting its inputs along a given axis, and calling each chunk on a different device.

    :param func: The function to be called.
    :type func: callable
    :param inputs: A list of inputs to pass into the function.
    :type inputs: sequence of arrays
    :param dev_strs: The gpu device strings, in the format "gpu:idx".
    :type dev_strs: int sequence of ints or sequence of strs
    :param input_axes: The axes along which to split each of the inputs, before passing to the function. Default is 0.
    :type input_axes: int or sequence of ints, optional
    :param output_axes: The axes along which to concat each of the returned outputs. Default is same as fist input axis.
    :type output_axes: int or sequence of ints, optional
    :param concat_output: Whether to concatenate each return values into a single array. Default is False.
    :type concat_output: bool, optional
    :return: The return from the function, following input splitting and re-concattenation across devices.
    """
    if isinstance(input_axes, int):
        input_axes = [input_axes]*len(inputs)
    if isinstance(dev_strs, int):
        dev_strs = ["gpu:{}".format(dev_strs)]
    elif isinstance(dev_strs[0], int):
        dev_strs = ["gpu:{}".format(i) for i in dev_strs]
    input_0 = inputs[0]
    start_dev = ivy.dev_str(input_0) if ivy.is_array(input_0) else input_0.dev_str
    dim_size = input_0.shape[input_axes[0]]
    num_chunks = len(dev_strs)
    chunk_size = dim_size / num_chunks
    chunk_size_rounded = int(np.round(chunk_size))
    chunk_size_diff = chunk_size - chunk_size_rounded
    total_diff = int(np.round(chunk_size_diff*num_chunks))
    chunk_sizes = [chunk_size_rounded]*num_chunks
    for i in range(np.abs(total_diff)):
        chunk_sizes[i] += np.sign(total_diff)
    inputs_split = [ivy.split(inp, chunk_sizes, input_axes[i], True) if ivy.is_array(inp)
                    else inp.split(chunk_sizes, input_axes[i], True) for i, inp in enumerate(inputs)]
    inputs_split_to_devs = [[ivy.to_dev(inp, d_str) if ivy.is_array(inp) else inp.to_dev(d_str)
                             for inp, d_str in zip(inps, dev_strs)] for inps in inputs_split]
    rets = [func(*inps, dev_str=dev_strs[i]) for i, inps in enumerate(zip(*inputs_split_to_devs))]
    # ToDo: make the line below more readable, there is a lot going on
    rets = [[ivy.to_dev(ret, start_dev) if ivy.is_array(ret) else
             (ret.to_dev(start_dev) if isinstance(ret, ivy.Container) else
              ([ivy.to_dev(r, start_dev) if ivy.is_array(r) else r.to_dev(start_dev)
                for r in ret] if isinstance(ret, (list, tuple)) else ret)) for ret in rts] for rts in rets]
    num_outputs = len(rets[0])
    if not concat_output:
        return [[r[i] for r in rets] for i in range(num_outputs)]
    if output_axes is None:
        output_axes = [input_axes[0]] * num_outputs
    elif isinstance(output_axes, int):
        output_axes = [output_axes] * num_outputs
    returns = list()
    ret0 = rets[0]
    # ToDo: possibly make this cleaner using list comprehension or recursion
    for i in range(num_outputs):
        if ivy.is_array(ret0[i]):
            returns.append(ivy.concatenate([r[i] for r in rets], output_axes[i]))
        elif isinstance(ret0[i], ivy.Container):
            returns.append(ivy.Container.concat([r[i] for r in rets], output_axes[i]))
        elif isinstance(ret0[i], (tuple, list)):
            ret0i_len = len(ret0[i])
            if ivy.is_array(ret0[i][0]):
                returns.append([ivy.concatenate([r[i][j] for r in rets], output_axes[i]) for j in range(ret0i_len)])
            elif isinstance(ret0[i][0], ivy.Container):
                returns.append([ivy.Container.concat([r[i][j] for r in rets], output_axes[i])
                                for j in range(ret0i_len)])
            else:
                returns.append([r[i] for r in rets])
        else:
            returns.append([r[i] for r in rets])
    return returns


def cache_fn(func):
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


def current_framework_str(f=None):
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


def einops_rearrange(x, pattern, **axes_lengths):
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


def einops_reduce(x, pattern, reduction, **axes_lengths):
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


def einops_repeat(x, pattern, **axes_lengths):
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


def get_min_denominator():
    """
    Get the global minimum denominator used by ivy for numerically stable division.
    """
    # noinspection PyProtectedMember
    return ivy._MIN_DENOMINATOR


def set_min_denominator(val):
    """
    Set the global minimum denominator used by ivy for numerically stable division.

    :param val: The new value to set the minimum denominator to.
    :type val: float
    """
    ivy._MIN_DENOMINATOR = val


def stable_divide(numerator, denominator, min_denominator=None):
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


def get_min_base():
    """
    Get the global minimum base used by ivy for numerically stable power raising.
    """
    # noinspection PyProtectedMember
    return ivy._MIN_BASE


def set_min_base(val):
    """
    Set the global minimum base used by ivy for numerically stable power raising.

    :param val: The new value to set the minimum base to.
    :type val: float
    """
    ivy._MIN_BASE = val


def stable_pow(base, exponent, min_base=None):
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
    if exponent < 1:
        return base ** exponent
    # noinspection PyProtectedMember
    return (base + default(min_base, ivy._MIN_BASE)) ** exponent
