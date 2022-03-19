"""
Collection of general Ivy functions.
"""

# global
import gc
import math
import einops
import inspect
import numpy as np
from numbers import Number
from typing import Callable, Any, Union, List, Tuple, Dict, Iterable, Optional

# local
import ivy
from ivy.functional.ivy.device import dev
from ivy.framework_handler import current_framework as _cur_framework

FN_CACHE = dict()
INF = float('inf')
TIMEOUT = 15.0
TMP_DIR = '/tmp'

















def equal(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]):
    """
    Computes the truth value of x1_i == x2_i for each element x1_i of the input array x1 with the respective
    element x2_i of the input array x2.

    :param x1: first input array. May have any data type.
    :param x2: second input array. Must be compatible with x1 (with Broadcasting). May have any data type.
    :return: an array containing the element-wise results. The returned array must have a data type of bool.
    """
    return _cur_framework(x1, x2).equal(x1, x2)





def shape(x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False)\
        -> Iterable[int]:
    """
    Returns the shape of the array x.

    :param x: Input array to infer the shape of.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :return: Shape of the array
    """
    return _cur_framework(x).shape(x, as_array)





def get_num_dims(x: Union[ivy.Array, ivy.NativeArray], as_array: bool = False) -> int:
    """
    Returns the number of dimensions of the array x.

    :param x: Input array to infer the number of dimensions for.
    :type x: array
    :param as_array: Whether to return the shape as a array, default False.
    :type as_array: bool, optional
    :return: Shape of the array
    """
    return _cur_framework(x).get_num_dims(x, as_array)


def minimum(x: Union[ivy.Array, ivy.NativeArray], y: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    :param x: Input array containing elements to minimum threshold.
    :type x: array
    :param y: Tensor containing minimum values, must be broadcastable to x.
    :type y: array
    :return: An array with the elements of x, but clipped to not exceed the y values.
    """
    return _cur_framework(x).minimum(x, y)


def maximum(x: Union[ivy.Array, ivy.NativeArray, Number], y: Union[ivy.Array, ivy.NativeArray, Number],
            ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    :param x: Input array containing elements to maximum threshold.
    :type x: array
    :param y: Tensor containing maximum values, must be broadcastable to x.
    :type y: array
    :return: An array with the elements of x, but clipped to not be lower than the y values.
    """
    return _cur_framework(x).maximum(x, y)


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





# noinspection PyShadowingBuiltins
def floor(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns element-wise largest integer not greater than x.

    :param x: Input array to floor.
    :type x: array
    :return: An array of the same shape and type as x, with the elements floored to integers.
    """
    return _cur_framework(x).floor(x)


# noinspection PyShadowingBuiltins
def abs(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the absolute value of each element in x.
    :param x: Input array containing elements to absolute value.
    :type x: array
    :return: A new array of the same shape as input array a, with all values now positive.
    """
    return _cur_framework(x).abs(x)


def argmin(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the index with the smallest value across axes of an array.

    :param x: Input array containing elements to argmin.
    :type x: array
    :param axis: Axis to perform the argmin, default is 0.
    :type axis: int, optional
    :return: Tensor containing the indices of the minimum values across the specified axis.
    """
    return _cur_framework(x).argmin(x, axis)


# noinspection PyShadowingNames
def arange(stop: Number, start: Number = 0, step: Number = 1, dtype: ivy.Dtype = None, dev: ivy.Device = None,
           ) -> Union[ivy.Array, ivy.NativeArray]:
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
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev: ivy.Device
    :return: Tensor of evenly spaced values.

            For floating point arguments, the length of the result is ceil((stop - start)/step).
            Because of floating point overflow, this rule may result in the last element of out being greater than stop.
    """
    return _cur_framework().arange(stop, start, step, dtype, dev)





def concatenate(xs: Iterable[Union[ivy.Array, ivy.NativeArray]], axis: int = -1)\
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
    return _cur_framework(xs[0]).concatenate(xs, axis)


def stack(xs: Iterable[Union[ivy.Array, ivy.NativeArray]], axis: int = 0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Joins a sequence of arrays along a new axis.
    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis: int = 0, it will be the first dimension and if axis: int = -1, it will be the last dimension.

    :param xs: Input arrays, each array must have the same shape.
    :type xs: sequence of arrays
    :param axis: The axis in the result array along which the input arrays are stacked.
    :type axis: int, optional
    :return: The stacked array has one more dimension than the input arrays.
    """
    return _cur_framework(xs[0]).stack(xs, axis)







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


def transpose(x: Union[ivy.Array, ivy.NativeArray], axes: Iterable[int] = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Permutes the dimensions of an array.

    :param x: Input array.
    :type x: array
    :param axes: By default, reverse the dimensions, otherwise permute the axes according to the values given.
    :type axes: sequence of ints of length N
    :return: x with its axes permuted.
    """
    return _cur_framework(x).transpose(x, axes)


def where(condition: Union[ivy.Array, ivy.NativeArray], x1: Union[ivy.Array, ivy.NativeArray],
          x2: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns elements chosen from x or y depending on condition.

    :param condition: Where True, yield x1, otherwise yield x2.
    :type condition: bool array
    :param x1: values from which to choose when condition is True.
    :type x1: array
    :param x2: values from which to choose when condition is False.
    :type x2: array
    :return: An array with elements from x1 where condition is True, and elements from x2 elsewhere.
    """
    return _cur_framework(x1).where(condition, x1, x2)


def indices_where(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns indices or true elements in an input boolean array.

    :param x: Boolean array, for which indices are desired.
    :type x: array
    :return: Indices for where the boolean array is True.
    """
    return _cur_framework(x).indices_where(x)


def reshape(x: Union[ivy.Array, ivy.NativeArray], newshape: Union[int, Iterable[int]])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gives a new shape to an array without changing its data.

    :param x: Tensor to be reshaped.
    :type x: array
    :param newshape: The new shape should be compatible with the original shape. One shape dimension can be -1.
                        In this case, the value is inferred from the length of the array and remaining dimensions.
    :type newshape: int or sequence of ints
    :return: Reshaped array.
    """
    return _cur_framework(x).reshape(x, newshape)


def broadcast_to(x: Union[ivy.Array, ivy.NativeArray], newshape: Iterable[int])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Broadcast the input tensor to newshape, adding dimensions of size 1 where the dimensions do not align.

    :param x: Tensor to be broadcast to new shape.
    :type x: array
    :param newshape: The new shape the tensor should be broadcast to.
    :type newshape: sequence of ints
    :return: Newly broadcast array.
    """
    return _cur_framework(x).broadcast_to(x, newshape)


def squeeze(x: Union[ivy.Array, ivy.NativeArray], axis: int = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Removes a single-dimensional entry from the shape of an array.

    :param x: Input data.
    :type x: array
    :param axis: Index for one of the single-dimensional entries in the shape.
                 If an axis is selected with shape entry greater than one, an error is raised.
    :type axis: int, optional
    :return: The input array, but with all (axis=None) or one (axis is int) of the dimensions of length 1 removed.
    """
    return _cur_framework(x).squeeze(x, axis)


def zeros_like(x: Union[ivy.Array, ivy.NativeArray], dtype: ivy.Dtype = None, dev: ivy.Device = None,
               ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns an array of zeros with the same shape and type as x, unless dtype provided which overrides.

    :param x: The shape and data-type of x define these same attributes of the returned array.
    :type x: array
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: ivy.Device, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype provided which overrides.
    """
    return _cur_framework(x).zeros_like(x, dtype, dev)


def ones(shape: Iterable[int], dtype: Union[ivy.Dtype, str] = 'float32', dev: ivy.Device = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: sequence of ints
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev: ivy.Device
    :return: Tensor of ones with the given shape and dtype.
    """
    return _cur_framework().ones(shape, dtype, dev)


# noinspection PyShadowingNames
def full(shape: Union[int, Tuple[int]], fill_value: Union[int, float], dtype: Optional[ivy.Dtype] = None,
         device: Optional[ivy.Device] = None):
    """
    Returns a new array having a specified shape and filled with fill_value.

    :param shape: output array shape.
    :param fill_value: fill value.
    :param dtype: output array data type.
    :param device: device on which to place the created array. Default: None.
    """
    return _cur_framework().full(shape, fill_value, dtype, device)


# noinspection PyShadowingNames
def one_hot(indices: Union[ivy.Array, ivy.NativeArray], depth: int, dev: ivy.Device = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a one-hot array
    :param indices: Indices for where the ones should be scattered *[batch_shape, dim]*
    :type indices: array
    :param depth: Scalar defining the depth of the one-hot dimension.
    :type depth: int
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: ivy.Device, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype provided which overrides.
    """
    return _cur_framework(indices).one_hot(indices, depth, dev)


def cross(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray])\
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
    :return: Vector cross product(s).
    """
    return _cur_framework(x1).cross(x1, x2)


def matmul(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Computes the matrix product of two arrays x1 and x2.

    :param x1: Input array 1.
    :type x1: array
    :param x2: Input array 2.
    :type x2: array
    :return: The matrix product of the input arrays.
    """
    return _cur_framework(x1).matmul(x1, x2)


def cumsum(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the cumulative sum of the elements along a given axis.

    :param x: Input array.
    :type x: array
    :param axis: Axis along which the cumulative sum is computed. By default 0.
    :type axis: int
    :return: Input array with cumulatively summed elements along axis.
    """
    return _cur_framework(x).cumsum(x, axis)


def cumprod(x: Union[ivy.Array, ivy.NativeArray], axis: int = 0, exclusive: bool = False)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the cumulative product of the elements along a given axis.

    :param x: Input array.
    :type x: array
    :param axis: Axis along which the cumulative product is computed. By default 0.
    :type axis: int
    :param exclusive: Whether to perform the cumprod exclusively. Defaults is False.
    :type exclusive: bool, optional
    :return: Input array with cumulatively multiplied elements along axis.
    """
    return _cur_framework(x).cumprod(x, axis, exclusive)


# noinspection PyShadowingNames
def identity(n: int, dtype: ivy.Dtype = 'float32', batch_shape: Iterable[int] = None, dev: ivy.Device = None,
             ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the identity array.
    The identity array is a square array with ones on the main diagonal.

    :param n: Number of rows (and columns) in n x n output.
    :type n: int
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                      Default is 'float32'.
    :type dtype: data-type string, optional
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev: ivy.Device
    :return: n x n array of type dtype, with its main diagonal set to one, and all other elements 0.
    """
    return _cur_framework().identity(n, dtype, batch_shape, dev)


def meshgrid(*xs: Iterable[Union[ivy.Array, ivy.NativeArray]], indexing: str = 'ij')\
        -> Iterable[Union[ivy.Array, ivy.NativeArray]]:
    """
    Broadcasts parameters for evaluation on an N-D grid.

    :param xs: input arrays
    :type xs: sequence of arrays
    :param indexing: The indexing method, either 'xy' or 'ij'. Default is 'ij'.
    :type indexing: str, optional
    :return: list of N-D coordinate arrays for evaluating expressions on an N-D grid
    """
    return _cur_framework().meshgrid(*xs, indexing=indexing)


# noinspection PyShadowingNames
def scatter_flat(indices: Union[ivy.Array, ivy.NativeArray], updates: Union[ivy.Array, ivy.NativeArray],
                 size: Optional[int] = None, tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
                 reduction: str = 'sum', dev: ivy.Device = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Scatter flat updates into a new flat array according to flat indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param size: The size of the result.
    :type size: int
    :param tensor: The tensor in which to scatter the results, default is None, in which case the size is used to
                    scatter into a zeros array.
    :param reduction: The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    :type reduction: str
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev: ivy.Device, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _cur_framework(indices).scatter_flat(indices, updates, size, tensor, reduction, dev)


# noinspection PyShadowingNames
def scatter_nd(indices: Union[ivy.Array, ivy.NativeArray], updates: Union[ivy.Array, ivy.NativeArray],
               shape: Optional[Iterable[int]] = None, tensor: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
               reduction: str = 'sum', dev: ivy.Device = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Scatter updates into a new array according to indices.

    :param indices: Indices for the new values to occupy.
    :type indices: array
    :param updates: Values for the new array to hold.
    :type updates: array
    :param shape: The shape of the result. Default is None, in which case tensor argument must be provided.
    :type shape: sequence of ints
    :param tensor: The tensor in which to scatter the results, default is None, in which case the shape arg is used to
                    scatter into a zeros array.
    :param reduction: The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    :type reduction: str
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as updates if None.
    :type dev: ivy.Device, optional
    :return: New array of given shape, with the values scattered at the indices.
    """
    return _cur_framework(indices).scatter_nd(indices, updates, shape, tensor, reduction, dev)


# noinspection PyShadowingNames
def gather(params: Union[ivy.Array, ivy.NativeArray], indices: Union[ivy.Array, ivy.NativeArray], axis: int = -1,
           dev: ivy.Device = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gather slices from params at axis according to indices.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param axis: The axis from which to gather from. Default is -1.
    :type axis: int, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: ivy.Device, optional
    :return: New array with the values gathered at the specified indices along the specified axis.
    """
    return _cur_framework(params).gather(params, indices, axis, dev)


# noinspection PyShadowingNames
def gather_nd(params: Union[ivy.Array, ivy.NativeArray], indices: Union[ivy.Array, ivy.NativeArray],
              dev: ivy.Device = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Gather slices from params into a array with shape specified by indices.

    :param params: The array from which to gather values.
    :type params: array
    :param indices: Index array.
    :type indices: array
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: ivy.Device, optional
    :return: New array of given shape, with the values gathered at the indices.
    """
    return _cur_framework(params).gather_nd(params, indices, dev)


def linear_resample(x: Union[ivy.Array, ivy.NativeArray], num_samples: int, axis: int = -1)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Performs linear re-sampling on input image.

    :param x: Input array
    :type x: array
    :param num_samples: The number of interpolated samples to take.
    :type num_samples: int
    :param axis: The axis along which to perform the resample. Default is last dimension.
    :type axis: int, optional
    :return: The array after the linear resampling.
    """
    return _cur_framework(x).linear_resample(x, num_samples, axis)















def multiprocessing(context: str = None):
    """
    Return framewrk-specific multi-processing module

    :param context: The context of the multiprocessing, either fork, forkserver or spawn. Default is None.
    :type context: str, optional
    :return: Multiprocessing module
    """
    return _cur_framework().multiprocessing(context)








def container_types():
    """
    Return all framework-specific types which should be hierarchically parsed in an ivy.Container. Such types must adopt
    a key-value structure, and exposes public methods .keys(), .values() and items().
    """
    # noinspection PyBroadException
    try:
        return _cur_framework().container_types()
    except ValueError:
        return []


def inplace_arrays_supported(f=None):
    """
    Determine whether inplace arrays are supported for the current backend framework.

    :return: Boolean, whether or not inplace arrays are supported.
    """
    return _cur_framework().inplace_arrays_supported()


def inplace_variables_supported(f=None):
    """
    Determine whether inplace variables are supported for the current backend framework.

    :return: Boolean, whether or not inplace variables are supported.
    """
    return _cur_framework().inplace_variables_supported()


def supports_inplace(x):
    """
    Determine whether inplace operations are supported for the data type of x.

    :param x: Input variable or array to check for inplace support for.
    :type x: variable or array
    :return: Boolean, whether or not inplace operations are supported for x.
    """
    if ivy.is_variable(x):
        return ivy.inplace_variables_supported()
    elif ivy.is_array(x):
        return ivy.inplace_arrays_supported()
    raise Exception('Input x must be either a variable or an array.')


def assert_supports_inplace(x):
    """
    Asserts that inplace operations are supported for x, else raises exception.

    :param x: Input variable or array to check for inplace support for.
    :type x: variable or array
    :return: True if support, raises exception otherwise
    """
    if not ivy.supports_inplace(x):
        raise Exception('Inplace operations are not supported {} types with {} backend'.format(
            type(x), ivy.current_framework_str()))
    return True


def inplace_update(x, val, f=None):
    """
    Perform in-place update for the input variable.

    :param x: The variable to update.
    :type x: variable
    :param val: The array to update the variable with.
    :type val: array
    :return: The variable following the in-place update.
    """
    return _cur_framework(x).inplace_update(x, val)


def inplace_decrement(x, val, f=None):
    """
    Perform in-place decrement for the input variable.

    :param x: The variable to decrement.
    :type x: variable
    :param val: The array to decrement the variable with.
    :type val: array
    :return: The variable following the in-place decrement.
    """
    return _cur_framework(x).inplace_decrement(x, val)


def inplace_increment(x, val, f=None):
    """
    Perform in-place increment for the input variable.

    :param x: The variable to increment.
    :type x: variable
    :param val: The array to increment the variable with.
    :type val: array
    :return: The variable following the in-place increment.
    """
    return _cur_framework(x).inplace_increment(x, val)
