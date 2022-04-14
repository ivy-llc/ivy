# global
from typing import Union, Tuple, Optional, List, Iterable
from numbers import Number

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def asarray(x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
             dtype: Optional[Union[ivy.Dtype, str]] = None,
             dev: Optional[Union[ivy.Device, str]] = None
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


array = asarray


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
       
       
    Examples:
    
    >>> shape = (3,5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    [[0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.]]
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


def full_like(x: Union[ivy.Array, ivy.NativeArray],
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


def zeros_like(x: Union[ivy.Array, ivy.NativeArray],
               dtype: Optional[Union[ivy.Dtype, str]] = None,
               dev: Optional[Union[ivy.Device, str]] = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
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


def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[ivy.Dtype] = None,
        device: Optional[ivy.Device] = None) \
        -> ivy.Array:
    """
    Returns a two-dimensional array with ones on the k h diagonal and zeros elsewhere.

    Parameters
    :param n_rows: number of rows in the output array.
    :param n_cols: number of columns in the output array. If None, the default number of columns in the output array is
                   equal to n_rows. Default: None.
    :param k: index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal,
              and 0 to the main diagonal. Default: 0.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default floating-
                  point data type. Default: None.
    :return: device on which to place the created array. Default: None.
    :return: an array where all elements are equal to zero, except for the k h diagonal, whose values are equal to one.
    """
    return _cur_framework().eye(n_rows, n_cols, k, dtype, device)

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

def meshgrid(*arrays: Union[ivy.Array, ivy.NativeArray], indexing: Optional[str] = 'xy') \
        -> List[ivy.Array]:
    """
    Returns coordinate matrices from coordinate vectors.
    Parameters
    ----------
    arrays: array
        an arbitrary number of one-dimensional arrays representing grid coordinates. Each array should have the same numeric data type.
    indexing: str
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases, respectively), the ``indexing`` keyword has no effect and should be ignored. Default: ``'xy'``.
    Returns
    -------
    out: List[array]
        list of N arrays, where ``N`` is the number of provided one-dimensional input arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional arrays having lengths ``Ni = len(xi)``,
        - if matrix indexing ``ij``, then each returned array must have the shape ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape ``(N2, N1, N3, ..., Nn)``.
        Accordingly, for the two-dimensional case with input one-dimensional arrays of length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M)``.
        Similarly, for the three-dimensional case with input one-dimensional arrays of length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M, P)``.
        Each returned array should have the same data type as the input arrays.
    """
    return _cur_framework().meshgrid(*arrays, indexing=indexing)


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


def from_dlpack(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.

    Parameters
    ----------
    x: object
        input (array) object.

    Returns
    -------
    out: array
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See :ref:`data-interchange` for details.
    """
    return _cur_framework(x).from_dlpack(x)


# Extra #
# ------#

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

