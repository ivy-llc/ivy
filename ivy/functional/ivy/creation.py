# For Review
# global
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence

import numpy as np

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    infer_device,
    infer_dtype,
    handle_out_argument,
    outputs_to_ivy_arrays,
    to_native_arrays_and_back,
    handle_nestable,
)


# Helpers #
# --------#


def _assert_fill_value_and_dtype_are_compatible(dtype, fill_value):
    assert (
        (ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype))
        and isinstance(fill_value, int)
    ) or (
        ivy.is_float_dtype(dtype)
        and isinstance(fill_value, float)
        or (isinstance(fill_value, bool))
    ), "the fill_value:\n\n{}\n\nand data type:\n\n{}\n\nare not compatible.".format(
        fill_value, dtype
    )


# Array API Standard #
# -------------------#


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
@handle_nestable
def arange(
    start: Number,
    /,
    stop: Optional[Number] = None,
    step: Number = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ceil((stop-start)/step) if stop - start and step have the
        same sign, and length 0 otherwise.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.arange.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().arange(
        start, stop, step, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@handle_nestable
def asarray(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
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
    copy
        boolean, indicating whether or not to copy the input. Default: ``None``.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array interpretation of x.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.asarray.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend().asarray(x, copy=copy, dtype=dtype, device=device)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def zeros(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing zeros.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.zeros.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> shape = (3, 5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    ivy.array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    """
    return current_backend().zeros(shape, dtype=dtype, device=device, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def ones(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing ones.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.ones.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> shape = (2,2)
    >>> y = ivy.ones(shape)
    >>> print(y)
    ivy.array([[1., 1.],
           [1., 1.]])

    """
    return current_backend().ones(shape, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@infer_dtype
@handle_nestable
def full_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    fill_value: float,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array filled with ``fill_value`` and having the same ``shape`` as
    an input array ``x`` .

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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.full_like.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------
    With int datatype:
    
    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])
    
    >>> fill_value = 0.000123
    >>> x = ivy.ones(5)
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With float datatype:
    
    >>> x = ivy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With ivy.NativeArray input:
    
    >>> x = ivy.native_array([3.0, 8.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x,fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123])
    
    >>> x = ivy.native_array([[3., 8., 2.], [2., 8., 3.]])
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([[0.000123, 0.000123, 0.000123],
           [0.000123, 0.000123, 0.000123]])

    With ivy.Container input:
    
    >>> x = ivy.Container(a=ivy.array([1.2,2.2324,3.234]), \
                           b=ivy.array([4.123,5.23,6.23]))
    >>> fill_value = 15.0
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    {
        a: ivy.array([15., 15., 15.]),
        b: ivy.array([15., 15., 15.])
    }

    Instance Method Examples:
    ------------------------

    With ivy.Array input:
    
    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = x.full_like(fill_value)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    With ivy.Container input:
    
    >>> x = ivy.Container(a=ivy.array([1,2,3]), \
                           b=ivy.array([4,5,6]))
    >>> fill_value = 10
    >>> y = x.full_like(fill_value)
    >>> print(y)
    {
        a: ivy.array([10, 10, 10]),
        b: ivy.array([10, 10, 10])
    }
    """
    return current_backend(x).full_like(
        x, fill_value, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def ones_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as x and filled with ones.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.ones_like.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.array([[0, 1, 2],[3, 4, 5]])
    >>> y = ivy.ones_like(x)
    >>> print(y)
    ivy.array([[1, 1, 1],
           [1, 1, 1]])
    """
    return current_backend(x).ones_like(x, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def zeros_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``zeros``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.zeros_like.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With 'ivy.Array' input:

        >>> x1 = ivy.array([1, 2, 3, 4, 5, 6])
        >>> y1 = ivy.zeros_like(x1)
        >>> print(y1)
        ivy.array([0, 0, 0, 0, 0, 0])

        >>> x2 = ivy.array([[0, 1, 2],[3, 4, 5]], dtype = ivy.float32)
        >>> y2 = ivy.zeros_like(x2)
        >>> print(y2)
        ivy.array([[0., 0., 0.],
               [0., 0., 0.]])

        >>> x3 = ivy.array([3., 2., 1.])
        >>> y3 = ivy.ones(3)
        >>> ivy.zeros_like(x3, out=y3)
        ivy.array([0., 0., 0.])

    With 'ivy.NativeArray' input:

        >>> x1 = ivy.native_array([[3, 8, 2],[2, 8, 3]])
        >>> y1 = ivy.zeros_like(x1)
        >>> print(y1)
        ivy.array([[0, 0, 0],
               [0, 0, 0]])


        >>> x2 = ivy.native_array([3, 8, 2, 0, 0, 2])
        >>> y2 = ivy.zeros_like(x2, dtype=ivy.IntDtype('int32'), device=ivy.Device('cpu'))
        >>> print(y2)
        ivy.array([0, 0, 0, 0, 0, 0])

        # Array ``y2`` is now stored on the CPU.

    With 'ivy.Container' input:

        >>> x = ivy.Container(a=ivy.array([3, 2, 1]), b=ivy.array([8, 2, 3]))
        >>> y = ivy.zeros_like(x)
        >>> print(y)
        {
            a: ivy.array([0, 0, 0]),
            b: ivy.array([0, 0, 0])
        }

    Instance Method Examples
    -------------------

    With 'ivy.Array' input:

        >>> x = ivy.array([2, 3, 8, 2, 1])
        >>> y = x.zeros_like()
        >>> print(y)
        ivy.array([0, 0, 0, 0, 0])

    With 'ivy.Container' input:

        >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
        >>> y = x.zeros_like()
        >>> print(y)
        {
            a: ivy.array([0., 0.]),
            b: ivy.array([0., 0.])
        }
    """
    return current_backend(x).zeros_like(x, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def tril(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as x. All elements above the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.tril.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).tril(x, k, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def triu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.    *,
    k
        diagonal below which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: 0.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the upper triangular part(s). The returned array must have
        the same shape and data type as x. All elements below the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.triu.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).triu(x, k, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def empty(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an uninitialized array having a specified shape

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.empty.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend().empty(shape, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def empty_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as x and containing uninitialized data.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.empty_like.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).empty_like(x, dtype=dtype, device=device, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: Optional[int] = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        device on which to place the created array. Default: None.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.eye.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend().eye(
        n_rows, n_cols, k, batch_shape, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
def linspace(
    start: Union[ivy.Array, ivy.NativeArray, float],
    stop: Union[ivy.Array, ivy.NativeArray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.linspace.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(start).linspace(
        start, stop, num, axis, endpoint=endpoint, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_nestable
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

        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
        the `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.meshgrid.html>`_ # noqa
        in the standard.

        Both the description and the type hints above assumes an array input for simplicity,
        but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
        instances in place of any of the arguments.

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
    return current_backend().meshgrid(*arrays, indexing=indexing)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
@handle_nestable
def full(
    shape: Union[ivy.Shape, ivy.NativeShape],
    fill_value: Union[float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array where every element is equal to `fill_value`.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.full.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :code:`ivy.Shape` input:

    >>> shape = ivy.Shape((2,2))
    >>> fill_value = 8.6
    >>> x = ivy.full(shape, fill_value)
    >>> print(x)
    ivy.array([[8.6, 8.6],
               [8.6, 8.6]])

    With :code:`ivy.NativeShape` input:

    >>> shape = ivy.NativeShape((2, 2, 2))
    >>> fill_value = True
    >>> dtype = ivy.bool
    >>> device = ivy.Device('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[[True,  True],
                [True,  True]],
               [[True,  True],
                [True,  True]]])

    With :code:`ivy.NativeDevice` input:

    >>> shape = ivy.NativeShape((1, 2))
    >>> fill_value = 0.68
    >>> dtype = ivy.float64
    >>> device = ivy.NativeDevice('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[0.68, 0.68]])

    With :code:'ivy.Container' input:

    >>> shape = ivy.Container(a=ivy.NativeShape((2, 1)), b=ivy.Shape((2, 1, 2)))
    >>> fill_value = ivy.Container(a=0.99, b=False)
    >>> dtype = ivy.Container(a=ivy.float64, b=ivy.bool)
    >>> device = ivy.Container(a=ivy.NativeDevice('cpu'), b=ivy.Device('cpu'))
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    {
        a: ivy.array([[0.99],
                      [0.99]]),
        b: ivy.array([[[False, False]],
                      [[False, False]]])
    }


    """
    return current_backend().full(
        shape, fill_value, dtype=dtype, device=device, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def from_dlpack(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Returns a new array containing the data from another (array) object with a
    ``__dlpack__`` method.

    Parameters
    ----------
    x  object
        input (array) object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.from_dlpack.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).from_dlpack(x, out=out)


# Extra #
# ------#


array = asarray


def native_array(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
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
@infer_dtype
@infer_device
@handle_nestable
def logspace(
    start: Union[ivy.Array, ivy.NativeArray, int],
    stop: Union[ivy.Array, ivy.NativeArray, int],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: Optional[int] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(start).logspace(
        start, stop, num, base, axis, dtype=dtype, device=device, out=out
    )
