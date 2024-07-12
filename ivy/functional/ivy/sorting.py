# global
from typing import Union, Optional, Literal, List

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions


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
def argsort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x
        input array.
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        sort order. If ``True``, the returned indices sort ``x`` in descending order
        (by value). If ``False``, the returned indices sort ``x`` in ascending order
        (by value). Default: ``False``.
    stable
        sort stability. If ``True``, the returned indices must maintain the relative
        order of ``x`` values which compare as equal. If ``False``, the returned indices
        may or may not maintain the relative order of ``x`` values which compare as
        equal (i.e., the relative order of ``x`` values which compare as equal is
        implementation-dependent). Default: ``True``.
    out
        optional output array, for writing the result to. It must have the same shape
        as ``x``.

    Returns
    -------
    ret
        an array of indices. The returned array must have the same shape as ``x``. The
        returned array must have the default array index data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.argsort.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([3,1,2])
    >>> y = ivy.argsort(x)
    >>> print(y)
    ivy.array([1,2,0])

    >>> x = ivy.array([4,3,8])
    >>> y = ivy.argsort(x, descending=True)
    >>> print(y)
    ivy.array([2,0,1])

    >>> x = ivy.array([[1.5, 3.2], [2.3, 2.3]])
    >>> ivy.argsort(x, axis=0, descending=True, stable=False, out=x)
    >>> print(x)
    ivy.array([[1, 0], [0, 1]])

    >>> x = ivy.array([[[1,3], [3,2]], [[2,4], [2,0]]])
    >>> y = ivy.argsort(x, axis=1, descending=False, stable=True)
    >>> print(y)
    ivy.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([5,1,3]), b=ivy.array([[0, 3], [3, 2]]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([1, 2, 0]),
        b: ivy.array([[0, 1], [1, 0]])
    }

    >>> x = ivy.Container(a=ivy.array([[3.5, 5],[2.4, 1]]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([[0,1],[1,0]])
    }

    >>> x = ivy.Container(a=ivy.array([4,3,6]), b=ivy.array([[4, 5], [2, 4]]))
    >>> y = ivy.argsort(x, descending=True)
    >>> print(y)
    {
        a: ivy.array([2, 0, 1]),
        b: ivy.array([[1, 0], [1, 0]])
    }

    >>> x = ivy.Container(a=ivy.array([[1.5, 3.2],[2.3, 4]]),
    ...                   b=ivy.array([[[1,3],[3,2],[2,0]]]))
    >>> y = x.argsort(axis=-1, descending=True, stable=False)
    >>> print(y)
    {
        a: ivy.array([[1,0],[1,0]]),
        b: ivy.array([[[1,0],[0, 1],[0, 1]]])
    }
    """
    return ivy.current_backend(x).argsort(
        x, axis=axis, descending=descending, stable=stable, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a sorted copy of an array.

    Parameters
    ----------
    x
        input array
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        direction  The direction in which to sort the values
    stable
        sort stability. If ``True``,
        the returned indices must maintain the relative order of ``x`` values which
        compare as equal. If ``False``, the returned indices may or may not maintain the
        relative order of ``x`` values which compare as equal (i.e., the relative order
        of ``x`` values which compare as equal is implementation-dependent).
        Default: ``True``.
    out
        optional output array, for writing the result to. It must have the same shape
        as ``x``.

    Returns
    -------
    ret
        An array with the same dtype and shape as ``x``, with the elements sorted
        along the given `axis`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sort.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([7, 8, 6])
    >>> y = ivy.sort(x)
    >>> print(y)
    ivy.array([6, 7, 8])

    >>> x = ivy.array([[[8.9,0], [19,5]],[[6,0.3], [19,0.5]]])
    >>> y = ivy.sort(x, axis=1, descending=True, stable=False)
    >>> print(y)
    ivy.array([[[19. ,  5. ],[ 8.9,  0. ]],[[19. ,  0.5],[ 6. ,  0.3]]])

    >>> x = ivy.array([1.5, 3.2, 0.7, 2.5])
    >>> y = ivy.zeros(5)
    >>> ivy.sort(x, descending=True, stable=False, out=y)
    >>> print(y)
    ivy.array([3.2, 2.5, 1.5, 0.7])

    >>> x = ivy.array([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> ivy.sort(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1,  2.2,  3.3],
        [-6.6, -5.5, -4.4]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([8, 6, 6]),b=ivy.array([[9, 0.7], [0.4, 0]]))
    >>> y = ivy.sort(x, descending=True)
    >>> print(y)
    {
        a: ivy.array([8, 6, 6]),
        b: ivy.array([[9., 0.7], [0.4, 0.]])
    }

    >>> x = ivy.Container(a=ivy.array([3, 0.7, 1]),b=ivy.array([[4, 0.9], [0.6, 0.2]]))
    >>> y = ivy.sort(x, descending=False, stable=False)
    >>> print(y)
    {
        a: ivy.array([0.7, 1., 3.]),
        b: ivy.array([[0.9, 4.], [0.2, 0.6]])
    }
    """
    return ivy.current_backend(x).sort(
        x, axis=axis, descending=descending, stable=stable, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def msort(
    a: Union[ivy.Array, ivy.NativeArray, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a copy of an array sorted along the first axis.

    Parameters
    ----------
    a
        array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        sorted array of the same type and shape as a

    Examples
    --------
    >>> a = ivy.asarray([[8, 9, 6],[6, 2, 6]])
    >>> ivy.msort(a)
    ivy.array(
        [[6, 2, 6],
         [8, 9, 6]]
        )
    """
    return ivy.current_backend(a).msort(a, out=out)


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
def searchsorted(
    x: Union[ivy.Array, ivy.NativeArray],
    v: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Union[ivy.Array, ivy.NativeArray, List[int]]] = None,
    ret_dtype: Union[ivy.Dtype, ivy.NativeDtype] = ivy.int64,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the indices of the inserted elements in a sorted array.

    Parameters
    ----------
    x
        Input array. If `sorter` is None, then it must be sorted in ascending order,
        otherwise `sorter` must be an array of indices that sort it.
    v
        specific elements to insert in array x1
    side
        The specific elements' index is at the 'left' side or
        'right' side in the sorted array x1. If the side is 'left', the
        index of the first suitable location located is given. If
        'right', return the last such index.
    ret_dtype
        the data type for the return value, Default: ivy.int64,
        only integer data types is allowed.
    sorter
        optional array of integer indices that sort array x into ascending order,
        typically the result of argsort.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
         An array of insertion points.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> v = ivy.array([2])
    >>> y  = ivy.searchsorted(x, v)
    >>> print(y)
    ivy.array([1])

    >>> x = ivy.array([0, 1, 2, 3])
    >>> v = ivy.array([3])
    >>> y  = ivy.searchsorted(x, v, side='right')
    >>> print(y)
    ivy.array([4])

    >>> x = ivy.array([0, 1, 2, 3, 4, 5])
    >>> v = ivy.array([[3, 1], [10, 3], [-2, -1]])
    >>> y  = ivy.searchsorted(x, v)
    >>> print(y)
    ivy.array([[3, 1],
       [6, 3],
       [0, 0]])
    """
    return ivy.current_backend(x, v).searchsorted(
        x,
        v,
        side=side,
        sorter=sorter,
        out=out,
        ret_dtype=ret_dtype,
    )
