# global
from numbers import Number
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device,
    handle_backend_invalid,
)


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the indices of the maximum values along a specified axis. When
    the maximum value occurs multiple times, only the indices corresponding to
    the first occurrence are returned.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        maximum value of the flattened array. Default = None.
    keepdims
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the array.
    dtype
        Optional data type of the output array.
    select_last_index
        If this is set to True, the index corresponding to the
        last occurrence of the maximum value will be returned
    out
        If provided, the result will be inserted into this array. It should be of the
        appropriate shape and dtype.

    Returns
    -------
    ret
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the maximum value; otherwise, a non-zero-dimensional array
        containing the indices of the maximum values. The returned array must have be
        the default array index data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.argmax.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0., 1., -1.])
    >>> y = ivy.argmax(x)
    >>> print(y)
    ivy.array([1])

    >>> x = ivy.array([-0., 1., -1.])
    >>> z = ivy.zeros((1,3), dtype=ivy.int64)
    >>> ivy.argmax(x, out=z)
    >>> print(z)
    ivy.array(1)

    >>> x = ivy.array([[1., -0., -1.], [-2., 3., 2.]])
    >>> y = ivy.argmax(x, axis=1)
    >>> print(y)
    ivy.array([0, 1])

    >>> x = ivy.array([[4., 0., -1.], [2., -3., 6]])
    >>> y = ivy.argmax(x, axis=1, keepdims=True)
    >>> print(y)
    ivy.array([[0], [2]])

    >>> x = ivy.array([[4., 0., -1.], [2., -3., 6]])
    >>> y = ivy.argmax(x, axis=1, dtype=ivy.int64)
    >>> print(y, y.dtype)
    ivy.array([0, 2]) int64

    >>> x = ivy.array([[4., 0., -1.],[2., -3., 6], [2., -3., 6]])
    >>> z = ivy.zeros((3,1), dtype=ivy.int64)
    >>> y = ivy.argmax(x, axis=1, keepdims=True, out=z)
    >>> print(z)
    ivy.array([[0],[2],[2]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.argmax(x)
    >>> print(y)
    {
        a: ivy.array(2),
        b: ivy.array(2)
    }
    """
    return current_backend(x).argmax(
        x,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        select_last_index=select_last_index,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def argmin(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    select_last_index: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the indices of the minimum values along a specified axis. When
    the minimum value occurs multiple times, only the indices corresponding to
    the first occurrence are returned.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        minimum value of the flattened array. Default = None.
    keepdims
        if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default = False.
    dtype
        An optional output_dtype from: int32, int64. Defaults to int64.
    select_last_index
        If this is set to True, the index corresponding to the
        last occurrence of the maximum value will be returned.
    out
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the minimum value; otherwise, a non-zero-dimensional array
        containing the indices of the minimum values. The returned array must have the
        default array index data type.

    Returns
    -------
    ret
        Array containing the indices of the minimum values across the specified axis.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.argmin.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 1., -1.])
    >>> y = ivy.argmin(x)
    >>> print(y)
    ivy.array(2)


    >>> x = ivy.array([[0., 1., -1.],[-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis=1)
    >>> print(y)
    ivy.array([2, 0])

    >>> x = ivy.array([[0., 1., -1.],[-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis=1, keepdims=True)
    >>> print(y)
    ivy.array([[2],
           [0]])

    >>> x = ivy.array([[0., 1., -1.],[-2., 1., 2.],[1., -2., 0.]])
    >>> y= ivy.zeros((3,1), dtype=ivy.int64)
    >>> ivy.argmin(x, axis=1, keepdims=True, out=y)
    >>> print(y)
    ivy.array([[2],
           [0],
           [1]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.argmin(x)
    >>> print(y)
    {
        a: ivy.array(1),
        b: ivy.array(0)
    }
    """
    return current_backend(x).argmin(
        x,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        select_last_index=select_last_index,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device
def nonzero(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[Tuple[ivy.Array], ivy.Array]:
    """Return the indices of the array elements which are non-zero.

    .. note::
        If ``x`` has a complex floating-point data type, non-zero elements
        are those elements having at least one component (real or imaginary)
        which is non-zero.

    .. note::
        If ``x`` has a boolean data type, non-zeroelements are those elements
        which are equal to ``True``.

    Parameters
    ----------
    x
        input array. Must have a positive rank. If `x` is zero-dimensional, the function
        must raise an exception.
    as_tuple
        if True, the output is returned as a tuple of indices, one for each
        dimension of the input, containing the indices of the true elements in that
        dimension. If False, the coordinates are returned in a (N, ndim) array,
        where N is the number of true elements. Default = True.
    size
        if specified, the function will return an array of shape (size, ndim).
        If the number of non-zero elements is fewer than size, the remaining elements
        will be filled with fill_value. Default = None.
    fill_value
        when size is specified and there are fewer than size number of elements,
        the remaining elements in the output array will be filled with fill_value.
        Default = 0.

    Returns
    -------
    ret
        a tuple of `k` arrays, one for each dimension of `x` and each of size `n`
        (where `n` is the total number of non-zero elements), containing the indices of
        the non-zero elements in that dimension. The indices must be returned in
        row-major, C-style order. The returned array must have the default array index
        data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.nonzero.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 10, 15, 20, -50, 0])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([1, 2, 3, 4]),)

    >>> x = ivy.array([[1, 2], [-1, -2]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([0, 0, 1, 1]), ivy.array([0, 1, 0, 1]))

    >>> x = ivy.array([[0, 2], [-1, -2]])
    >>> y = ivy.nonzero(x, as_tuple=False)
    >>> print(y)
    ivy.array([[0, 1], [1, 0], [1, 1]])

    >>> x = ivy.array([0, 1])
    >>> y = ivy.nonzero(x, size=2, fill_value=4)
    >>> print(y)
    (ivy.array([1, 4]),)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[10, 20], [10, 0], [0, 0]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([0, 0, 1]), ivy.array([0, 1, 0]))

    >>> x = ivy.native_array([[0], [1], [1], [0], [1]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([1, 2, 4]), ivy.array([0, 0, 0]))

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0,1,2,3,0]), b=ivy.array([1,1, 0,0]))
    >>> y = ivy.nonzero(x)
    >>> print(y)
    [{
        a: ivy.array([1, 2, 3]),
        b: ivy.array([0, 1])
    }]

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    With :class:`ivy.Array` instance method:

    >>> x = ivy.array([0,0,0,1,1,1])
    >>> y = x.nonzero()
    >>> print(y)
    (ivy.array([3, 4, 5]),)

    With :class:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1,1,1]), b=ivy.native_array([0]))
    >>> y = x.nonzero()
    >>> print(y)
    [{
        a: ivy.array([0, 1, 2]),
        b: ivy.array([])
    }]
    """
    return current_backend(x).nonzero(
        x, as_tuple=as_tuple, size=size, fill_value=fill_value
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def where(
    condition: Union[ivy.Array, ivy.NativeArray],
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition
        Where True, yield x1, otherwise yield x2.
    x1
        values from which to choose when condition is True.
    x2
        values from which to choose when condition is False.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with elements from x1 where condition is True, and elements from x2
        elsewhere.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.where.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> condition = ivy.array([[True, False], [True, True]])
    >>> x1 = ivy.array([[1, 2], [3, 4]])
    >>> x2 = ivy.array([[5, 6], [7, 8]])
    >>> res = ivy.where(condition, x1, x2)
    >>> print(res)
    ivy.array([[1, 6],
           [3, 4]])

    >>> x1 = ivy.array([[6, 13, 22, 7, 12], [7, 11, 16, 32, 9]])
    >>> x2 = ivy.array([[44, 20, 8, 35, 9], [98, 23, 43, 6, 13]])
    >>> res = ivy.where(((x1 % 2 == 0) & (x2 % 2 == 1)), x1, x2)
    >>> print(res)
    ivy.array([[44, 20,  8, 35, 12],
           [98, 23, 16,  6, 13]])

    With :class:`ivy.Container` input:

    >>> x1 = ivy.Container(a=ivy.array([3, 1, 5]), b=ivy.array([2, 4, 6]))
    >>> x2 = ivy.Container(a=ivy.array([0, 7, 2]), b=ivy.array([3, 8, 5]))
    >>> condition = x1.a > x2.a
    >>> res = x1.where(condition, x2)
    >>> print(res)
    {
        a: ivy.array([1, 0, 1]),
        b: ivy.array([1, 0, 1])
    }
    """
    return current_backend(x1).where(condition, x1, x2, out=out)


# Extra #
# ------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def argwhere(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the indices of all non-zero elements of the input array.

    Parameters
    ----------
    x
        input array, for which indices are desired.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Indices of non-zero elements.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[1, 2], [3, 4]])
    >>> res = ivy.argwhere(x)
    >>> print(res)
    ivy.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    >>> x = ivy.array([[0, 2], [3, 4]])
    >>> res = ivy.argwhere(x)
    >>> print(res)
    ivy.array([[0, 1], [1, 0], [1, 1]])

    >>> x = ivy.array([[0, 2], [3, 4]])
    >>> y = ivy.zeros((3, 2), dtype=ivy.int64)
    >>> res = ivy.argwhere(x, out=y)
    >>> print(res)
    ivy.array([[0, 1], [1, 0], [1, 1]])

    With a :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2]), b=ivy.array([3, 4]))
    >>> res = ivy.argwhere(x)
    >>> print(res)
    {
        a: ivy.array([[0], [1]]),
        b: ivy.array([[0], [1]])
    }

    >>> x = ivy.Container(a=ivy.array([1, 0]), b=ivy.array([3, 4]))
    >>> res = ivy.argwhere(x)
    >>> print(res)
    {
        a: ivy.array([[0]]),
        b: ivy.array([[0], [1]])
    }
    """
    return current_backend(x).argwhere(x, out=out)
