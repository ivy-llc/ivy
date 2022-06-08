# global
import numpy as np
from numbers import Number
from typing import Union, Tuple, Optional, List

# local
import ivy
from ivy.backend_handler import current_backend as _cur_backend
from ivy.func_wrapper import (
    infer_device,
    infer_dtype,
    handle_out_argument,
    outputs_to_ivy_arrays,
    to_native_arrays_and_back,
)


# Array API Standard #
# -------------------#


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
def arange(
    start: Number,
    stop: Optional[Number] = None,
    step: Number = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Returns evenly spaced values within a given interval, with the spacing being
    specified.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop). For integer arguments the function
    is equivalent to the Python built-in range function, but returns an array in the
    chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    Parameters
    ----------
    start
        if stop is specified, the start of interval (inclusive); otherwise, the end of
        the interval (exclusive). If stop is not specified, the default starting value
        is 0.
    stop
        the end of the interval. Default: None.
    step
        the distance between two adjacent elements (out[i+1] - out[i]). Must not be 0;
        may be negative, this results in an empty array if stop >= start. Default: 1.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from start, stop and step. If those are all integers, the output array
        dtype must be the default integer dtype; if one or more have type float, then
        the output array dtype must be the default floating-point data type. Default:
        None.
    device
        device on which to place the created array. Default: None.

    Returns
    -------
    ret
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ceil((stop-start)/step) if stop - start and step have the
        same sign, and length 0 otherwise.

    """
    return _cur_backend().arange(start, stop, step, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_device
def asarray(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Converts the input to an array.

    Parameters
    ----------
    x
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype
        datatype, optional. Datatype is inferred from the input data.
    device
        device on which to place the created array. Default: None.

    Returns
    -------
    ret
        An array interpretation of x.

    """
    return _cur_backend().asarray(x, copy=copy, dtype=dtype, device=device)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
def zeros(
    shape: Union[int, Tuple[int], List[int]],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an array containing zeros.

    Examples
    --------
    >>> shape = (3, 5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    ivy.array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])

    """
    return _cur_backend().zeros(shape, dtype=dtype, device=device)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
def ones(
    shape: Union[int, Tuple[int], List[int]],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default  ``None``.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an array containing ones.

    Examples
    --------
    >>> shape = (2,2)
    >>> y = ivy.ones(shape)
    >>> print(y)
    ivy.array([[1.,  1.],
               [1.,  1.]])

    """
    return _cur_backend().ones(shape, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
def full_like(
    x: Union[ivy.Array, ivy.NativeArray],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array filled with ``fill_value`` and having the same ``shape`` as
    an input array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    fill_value
        Scalar fill value
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    """
    return _cur_backend(x).full_like(x, fill_value, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
def ones_like(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array filled with ones and having the same shape as an input array
    ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from x. Default  ``None``.
    device
        device on which to place the created array. If device is ``None``, the output
        array device must be inferred from x. Default: ``None``.

    Returns
    -------
    ret
        an array having the same shape as x and filled with ones.

    Examples
    --------
    >>> x = ivy.array([[0, 1, 2],[3, 4, 5]])
    >>> y = ivy.ones_like(x)
    >>> print(y)
    ivy.array([[1, 1, 1],[1, 1, 1]])

    """
    return _cur_backend(x).ones_like(x, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
def zeros_like(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array filled with zeros and having the same ``shape`` as an input
    array ``x``.

    Parameters
    ----------
    x
         input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``zeros``.

    Examples
    --------
    >>> x = ivy.array([[0, 1, 2],[3, 4, 5]])
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    ivy.array([[0, 0, 0],
               [0, 0, 0]])

    """
    return _cur_backend(x).zeros_like(x, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
def tril(x: Union[ivy.Array, ivy.NativeArray], k: int = 0) -> ivy.Array:
    """Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.
    k
        diagonal above which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    ret
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as x. All elements above the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.

    """
    return _cur_backend(x).tril(x, k)


@to_native_arrays_and_back
@handle_out_argument
def triu(x: Union[ivy.Array, ivy.NativeArray], k: int = 0) -> ivy.Array:
    """Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.
    k
        diagonal below which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    ret
        an array containing the upper triangular part(s). The returned array must have
        the same shape and data type as x. All elements below the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.

    """
    return _cur_backend(x).triu(x, k)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
def empty(
    shape: Union[int, Tuple[int], List[int]],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: None.
    device
        device on which to place the created array. Default: None.

    Returns
    -------
    ret
        an uninitialized array having a specified shape

    """
    return _cur_backend().empty(shape, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
def empty_like(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns an uninitialized array with the same shape as an input array x.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from x. Default  None.
    device
        device on which to place the created array. If device is None, the output array
        device must be inferred from x. Default: None.

    Returns
    -------
    ret
        an array having the same shape as x and containing uninitialized data.

    """
    return _cur_backend(x).empty_like(x, dtype=dtype, device=device)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a two-dimensional array with ones on the k diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows
        number of rows in the output array.
    n_cols
        number of columns in the output array. If None, the default number of columns in
        the output array is equal to n_rows. Default: None.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and 0 to the main diagonal. Default: 0.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: None.
    device
         device on which to place the created array.

    Returns
    -------
    ret
        device on which to place the created array. Default: None.

    """
    return _cur_backend().eye(n_rows, n_cols, k, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
def linspace(
    start: Union[ivy.Array, ivy.NativeArray, int],
    stop: Union[ivy.Array, ivy.NativeArray, int],
    num: int,
    axis: int = None,
    endpoint: bool = True,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Generates a certain number of evenly-spaced values in an interval along a given
    axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in
    an interval.

    Parameters
    ----------
    start
        First entry in the range.
    stop
        Final entry in the range.
    num
        Number of values to generate.
    axis
        Axis along which the operation is performed.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.

    """
    return _cur_backend(start).linspace(
        start, stop, num, axis, endpoint=endpoint, dtype=dtype, device=device
    )


@to_native_arrays_and_back
def meshgrid(
    *arrays: Union[ivy.Array, ivy.NativeArray], indexing: Optional[str] = "xy"
) -> List[ivy.Array]:
    """Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    indexing
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and should be ignored.
        Default: ``'xy'``.

    Returns
    -------
    ret
        list of N arrays, where ``N`` is the number of provided one-dimensional input
        arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional
        arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape
          ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape
          ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of
        length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must
        have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned
        array must have shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of
        length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned
        array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then
        each returned array must have shape ``(N, M, P)``.

        Each returned array should have the same data type as the input arrays.


        This method conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
        the `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.meshgrid.htm>`_  # noqa
        in the standard. The descriptions above assume an array input for simplicity,
        but the method also accepts :code:`ivy.Array` or :code:`ivy.NativeArray`
        instances, as shown in the type hints and also the examples below.

        Functional Examples
        -------------------

        With :code:`ivy.Array` input:

        >>> x = ivy.array([1, 2])
        >>> y = ivy.array([3, 4])
        >>> xv, yv = ivy.meshgrid(x, y)
        >>> print(xv)
        ivy.array([[1, 2],
                   [1, 2]])
        >>> print(yv)
        ivy.array([[3, 3],
                   [4, 4]])

        >>> x = ivy.array([1, 2, 5])
        >>> y = ivy.array([4, 1])
        >>> xv, yv = ivy.meshgrid(x, y, indexing='ij')
        >>> print(xv)
        ivy.array([[1, 1],
                   [2, 2],
                   [5, 5]])
        >>> print(yv)
        ivy.array([[4, 1],
                   [4, 1],
                   [4, 1]])

        With :code:`ivy.NativeArray` input:

        >>> x = ivy.native_array([1, 2])
        >>> y = ivy.native_array([3, 4])
        >>> xv, yv = ivy.meshgrid(x, y)
        >>> print(xv)
        ivy.array([[1, 2],
                   [1, 2]])
        >>> print(yv)
        ivy.array([[3, 3],
                   [4, 4]])

    """
    return _cur_backend().meshgrid(*arrays, indexing=indexing)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape
        output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``fill_value``. If the fill value is an ``int``, the output
        array data type must be the default integer data type. If the fill value is a
        ``float``, the output array data type must be the default floating-point data
        type. If the fill value is a ``bool``, the output array must have boolean data
        type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an array where every element is equal to `fill_value`.

    Examples
    --------
    >>> shape = (2,2)
    >>> fill_value = 10
    >>> y = ivy.full(shape, fill_value)
    >>> print(y)
    ivy.array([[10., 10.],
               [10., 10.]])

    """
    return _cur_backend().full(shape, fill_value, dtype=dtype, device=device)


@to_native_arrays_and_back
@handle_out_argument
def from_dlpack(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """Returns a new array containing the data from another (array) object with a
    ``__dlpack__`` method.

    Parameters
    ----------
    x  object
        input (array) object.

    Returns
    -------
    ret
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.

    """
    return _cur_backend(x).from_dlpack(x)


# Extra #
# ------#

array = asarray


def native_array(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.NativeArray:
    """Converts the input to a native array.

    Parameters
    ----------
    x
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype
        datatype, optional. Datatype is inferred from the input data.
    device
        device on which to place the created array. Default: None.

    Returns
    -------
    ret
        A native array interpretation of x.

    """
    # ToDo: Make this more efficient,
    # ideally without first converting to ivy.Array with ivy.asarray and then
    # converting back to native with ivy.to_native

    return ivy.to_native(ivy.asarray(x, dtype=dtype, device=device))


@to_native_arrays_and_back
@handle_out_argument
@infer_device
def logspace(
    start: Union[ivy.Array, ivy.NativeArray, int],
    stop: Union[ivy.Array, ivy.NativeArray, int],
    num: int,
    base: float = 10.0,
    axis: int = None,
    *,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Generates a certain number of evenly-spaced values in log space, in an interval
    along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in
    an interval.

    Parameters
    ----------
    start
        First entry in the range.
    stop
        Final entry in the range.
    num
        Number of values to generate.
    base
        The base of the log space. Default is 10.0
    axis
        Axis along which the operation is performed.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.

    """
    return _cur_backend(start).logspace(start, stop, num, base, axis, device=device)
