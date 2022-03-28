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
    return _cur_framework(x).matrix_transpose(x, axes)


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






