# global
from typing import Union, Tuple, Optional, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def asarray(x: Union[ivy.Array, ivy.NativeArray],
             dtype: Optional[Union[ivy.Dtype, str]] = None,
             dev: Optional[Union[ivy.Device, str]] = None,
             ) -> ivy.Array:
     """
     Converts the input to an array.

     Parameters
     ----------
     x:
         input data, in any form that can be converted to an array.
         This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.

     dtype:
         datatype, optional. Datatype is inferred from the input data.

     dev:
         device on which to place the created array. Default: None.

     Returns
     --------
     An array interpretation of x.
     """
     return _cur_framework(x).asarray(x, dtype, dev)


def zeros(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[ivy.Dtype] = None,
          device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape:
       output array shape.
    dtype:
       output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device:
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    out:
       an array containing zeros.

    """
    return _cur_framework().zeros(shape, dtype, device)


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[ivy.Dtype] = None,
         device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape:
        output array shape.
    dtype:
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device:
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out:
        an array containing ones.
    """
    return _cur_framework().ones(shape, dtype, device)


def full_like(x: Union[ivy.Array, ivy.NativeArray], /,
              fill_value: Union[int, float], *,
              dtype: Optional[Union[ivy.Dtype, str]] = None,
              device: Optional[Union[ivy.Device, str]] = None,
              ) -> ivy.Array:
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.

    Parameters
    x:input array from which to derive the output array shape.

    fill_value: Scalar fill value

    dtype:output array data type.
    If dtype is None, the output array data type must be inferred from x.
    Default: None.

    device:device on which to place the created array.
    If device is None,the output array device must be inferred from x.
    Default: None.

    Returns
    out:an array having the same shape as x and where every element is equal to fill_value.
    """
    return _cur_framework(x).full_like(x, fill_value, dtype=dtype, device=device)

def ones_like( x: Union[ivy.Array, ivy.NativeArray],
              dtype: Optional[Union[ivy.Dtype, str]] = None,
              dev: Optional[Union[ivy.Device, str]] = None,
              ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a new array filled with ones and having the same shape as an input array x.

    :param x: Input array from which to derive the output array shape.
    :param dtype: Output array data type. If dtype is None, the output array data type must be inferred from x.
    Default: None.
    :param dev: device on which to place the created array. If device is None, the output array device must be inferred from x.
    Default: None.
    :return: An array having the same shape as x and filled with ones.
    """
    return _cur_framework(x).ones_like(x, dtype, dev)


def tril(x: Union[ivy.Array, ivy.NativeArray],
         k: int = 0) \
        -> ivy.Array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) x.

    Parameters
    ----------
    x:
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
    k:
        diagonal above which to zero elements. If k = 0, the diagonal is the main diagonal. If k < 0, the diagonal is
        below the main diagonal. If k > 0, the diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    out:
        an array containing the lower triangular part(s). The returned array must have the same shape and data type as
        x. All elements above the specified diagonal k must be zeroed. The returned array should be allocated on the
        same device as x.
    """
    return _cur_framework(x).tril(x, k)


def triu(x: Union[ivy.Array, ivy.NativeArray],
         k: int = 0) \
         -> ivy.Array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) x.

    Parameters
    ----------
    x:
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
    k:
        diagonal below which to zero elements. If k = 0, the diagonal is the main diagonal. If k < 0, the diagonal is 
        below the main diagonal. If k > 0, the diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    out:
        an array containing the upper triangular part(s). The returned array must have the same shape and data type as 
        x. All elements below the specified diagonal k must be zeroed. The returned array should be allocated on the 
        same device as x.
    """
    return _cur_framework(x).triu(x, k)
    

def empty(shape: Union[int, Tuple[int],List[int]],
          dtype: Optional[ivy.Dtype] = None,
          device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.
    :param shape: output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default
                  floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :return: an uninitialized array having a specified shape
    """
    return _cur_framework().empty(shape, dtype, device)


def empty_like(x: Union[ivy.Array, ivy.NativeArray], 
               dtype: Optional[Union[ivy.Dtype, str]] = None, 
               dev: Optional[Union[ivy.Device, str]] = None)\
        -> ivy.Array:
    """
    Returns an uninitialized array with the same shape as an input array x.

    :param x:  input array from which to derive the output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be inferred from x. Default: None.
    :param dev: device on which to place the created array. If device is None, the output array device must be inferred from x. Default: None.
    :return: an array having the same shape as x and containing uninitialized data.
    """
    return _cur_framework(x).empty_like(x, dtype, dev)


# noinspection PyShadowingNames
def linspace(start: Union[ivy.Array, ivy.NativeArray, int], stop: Union[ivy.Array, ivy.NativeArray, int],
             num: int, axis: int = None, dev: ivy.Device = None)\
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
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev: ivy.Device
    :return: Tensor of evenly-spaced values.
    """
    return _cur_framework(start).linspace(start, stop, num, axis, dev)


# Extra #
# ------#
# noinspection PyShadowingNames
def array(object_in: Union[List, ivy.Array, ivy.NativeArray], dtype: Union[ivy.Dtype, str] = None,
          dev: ivy.Device = None) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Creates an array.

    :param object_in: An array_like object, which exposes the array interface,
            an object whose __array__ method returns an array, or any (nested) sequence.
    :type object_in: array
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
        If not given, then the type will be determined as the minimum type required to hold the objects in the
        sequence.
    :type dtype: data-type string, optional
    :param dev: device string on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type dev: ivy.Device
    :return: An array object satisfying the specified requirements, in the form of the selected framework.
    """
    return _cur_framework(object_in).array(object_in, dtype, dev)


# noinspection PyShadowingNames
def logspace(start: Union[ivy.Array, ivy.NativeArray, int], stop: Union[ivy.Array, ivy.NativeArray, int],
             num: int, base: float = 10., axis: int = None, dev: ivy.Device = None)\
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
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    :type dev: ivy.Device
    :return: Tensor of evenly-spaced values.
    """
    return _cur_framework(start).logspace(start, stop, num, base, axis, dev)


