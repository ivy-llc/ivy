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
)
from ivy.utils.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def argsort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Return the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x:  Union[ivy.Array, ivy.Container]
        input array.
    axis: int, optional
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending: bool, optional
        sort order. If ``True``, the returned indices sort ``x`` in descending order
        (by value). If ``False``, the returned indices sort ``x`` in ascending order
        (by value). Default: ``False``.
    stable: bool, optional
        sort stability. If ``True``, the returned indices must maintain the relative
        order of ``x`` values which compare as equal. If ``False``, the returned indices
        may or may not maintain the relative order of ``x`` values which compare as
        equal (i.e., the relative order of ``x`` values which compare as equal is
        implementation-dependent). Default: ``True``.
    out: Union[ivy.Array, None], optional
        optional output array, for writing the result to. It must have the same shape
        as ``x``.

    Returns
    -------
    ret: Union[ivy.Array, ivy.Container]
        an array of indices. The returned array must have the same shape as ``x``. The
        returned array must have the default array index data type.

    Raises
    -------
    TypeError
        If ``x`` is not an instance of :class:`ivy.Array` or :class:`ivy.NativeArray`.
        If ``out`` is not ``None`` and is not an instance of :class:`ivy.Array`.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.sorting_functions.argsort.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments

    Examples
    --------
    With :class:`ivy.Array` input:

    Example 1: Sorting x in ascending order and returning the indices:

    >>> x = ivy.array([3,1,2])
    >>> y = ivy.argsort(x)
    >>> print(y)
    ivy.array([1, 2, 0])

    Example 2: Sorting x in descending order and returning the indices:

    >>> x = ivy.array([4,3,8])
    >>> y = ivy.argsort(x, descending=True)
    >>> print(y)
    ivy.array([2, 0, 1])

    Example 3: Sorting x along axis 0 in descending order, in an unstable manner, and storing the indices in x:

    >>> x = ivy.array([[1.5, 3.2], [2.3, 2.3]])
    >>> ivy.argsort(x, axis=0, descending=True, stable=False, out=x)
    >>> print(x)
    ivy.array([[1, 0],
            [0, 1]])


    Example 4: Sorting x along axis 1 in ascending order, in a stable manner:

    >>> x = ivy.array([[[1,3], [3,2]], [[2,4], [2,0]]])
    >>> y = ivy.argsort(x, axis=1, descending=False, stable=True)
    >>> print(y)
    ivy.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])

    Example 5: Sort x along axis 0 in ascending order:
    >>> x = ivy.array([[1, 3], [4, 2], [0, 1]])
    >>> y = ivy.argsort(x, axis=0)
    >>> print(y)
    ivy.array([[2, 2],
            [0, 1],
            [1, 0]])



    With :class:`ivy.Container` input:

    Example 6: Sorting the elements of the container and returning their indices.
    The resulting container y has the same structure as x, but with each element replaced by its sorted indices.

    >>> x = ivy.Container(a=ivy.array([5,1,3]), b=ivy.array([[0, 3], [3, 2]]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([1, 2, 0]),
        b: ivy.array([[0, 1],
                      [1, 0]])
    }

    Example 7: Sorting the elements of the container and returning their indices.
    The resulting container y also has only one element, which is the indices that would sort the input array.

    >>> x = ivy.Container(a=ivy.array([[3.5, 5],[2.4, 1]]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([[0,1],[1,0]])
    }

    Example 8: Setting the descending argument to True to sort the elements in descending order.
    The resulting container y has the same structure as x, but with each element replaced by its sorted indices in descending order.

    >>> x = ivy.Container(a=ivy.array([4,3,6]), b=ivy.array([[4, 5], [2, 4]]))
    >>> y = ivy.argsort(x, descending=True)
    >>> print(y)
    {
        a: ivy.array([2, 0, 1]),
        b: ivy.array([[1, 0], [1, 0]])
    }

    Example 9: Sorting a container with two arrays a and b, the arrays are sorted along the last dimension (axis=-1) in descending order (descending=True).
    The resulting sorted arrays are returned as a container y, where the keys correspond to the keys of the input container x.

    >>> x = ivy.Container(a=ivy.array([[1.5, 3.2],[2.3, 4]]), b=ivy.array([[[1,3],[3,2],[2,0]]]))
    >>> y = x.argsort(axis=-1, descending=True, stable=False)
    >>> print(y)
    {
        a: ivy.array([[1,0],[1,0]]),
        b: ivy.array([[[1,0],[0, 1],[0, 1]]])
    }

    Example 10: Sorting a container with two arrays a, b and c, the arrays are sorted along the last dimension (axis=-1) in ascending order.
    The resulting sorted arrays are returned as a container y, where the keys correspond to the keys of the input container x.

    >>> x = ivy.Container(a=ivy.array([5, 1, 3]), b=ivy.array([[0, 3], [3, 2]]), c=ivy.array([9, 7, 6]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([1, 2, 0]),
        b: ivy.array([[0, 1],
                      [1, 0]]),
        c: ivy.array([2, 1, 0])
    }
    """
    return ivy.current_backend(x).argsort(
        x, axis=axis, descending=descending, stable=stable, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x:  array-like or ivy.Container
        Input array or container.
    axis:   int, optional
            axis along which to sort. If set to ``-1``, the function must sort along the
            last axis. Default: ``-1``.
    descending: bool, optional
                direction  The direction in which to sort the values
    stable: bool, optional
            sort stability. If ``True``,
            the returned indices must maintain the relative order of ``x`` values which
            compare as equal. If ``False``, the returned indices may or may not maintain the
            relative order of ``x`` values which compare as equal (i.e., the relative order
            of ``x`` values which compare as equal is implementation-dependent).
            Default: ``True``.
    out:    array-like or ivy.Container, optional
             Optional output array or container, for writing the result to. It must have the
            same shape as ``x``. Default: ``None``.

    Returns
    -------
    array:
        An array with the same dtype and shape as ``x``, with the elements sorted
        along the given `axis`.

    Raises
    -------
    TypeError:
        If the input is not an array-like or ivy.Container instance.
    ValueError:
        If the `axis` parameter is invalid or out of range.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.sorting_functions.sort.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments


    Examples
    --------
    With :class:`ivy.Array` input:

    Example 1: Sorting a 1D array in ascending order. The resulting array is printed.

    >>> x = ivy.array([7, 8, 6])
    >>> y = ivy.sort(x)
    >>> print(y)
    ivy.array([6, 7, 8])

    Example 2: Sorting a 3D array in descending order along axis 1.
    The stable argument is set to False, which means that equal elements are not guaranteed to maintain their original order.

    >>> x = ivy.array([[[8.9,0], [19,5]],[[6,0.3], [19,0.5]]])
    >>> y = ivy.sort(x, axis=1, descending=True, stable=False)
    >>> print(y)
    ivy.array([[[19. ,  5. ],[ 8.9,  0. ]],[[19. ,  0.5],[ 6. ,  0.3]]])


    Example 3: Sorting a 1D array in descending order, and writes the result to an existing array y using the out argument.
    >>> x = ivy.array([1.5, 3.2, 0.7, 2.5])
    >>> y = ivy.zeros(5)
    >>> ivy.sort(x, descending=True, stable=False, out=y)
    >>> print(y)
    ivy.array([3.2, 2.5, 1.5, 0.7])

    Example 4: Sorting a 2D array in ascending order, and writes the result to the same array x using the out argument

    >>> x = ivy.array([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> ivy.sort(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1,  2.2,  3.3],
        [-6.6, -5.5, -4.4]])

    Example 5: Sorting a 3D array in ascending order along axis 0. The stable argument is set to True.The resulting array is printed using print.

    >>> x = ivy.array([[[1, 3], [3, 2]], [[2, 4], [2, 0]]])
    >>> y = ivy.sort(x, axis=0, descending=False, stable=True)
    >>> print(y)
    ivy.array([[[1, 3], [2, 0]], [[2, 4], [3, 2]]])

    With :class:`ivy.Container` input:

    Example 6: Sorting a container x has two arrays, a and b, in descending order and returns the sorted container y.

    >>> x = ivy.Container(a=ivy.array([8, 6, 6]),b=ivy.array([[9, 0.7], [0.4, 0]]))
    >>> y = ivy.sort(x, descending=True)
    >>> print(y)
    {
        a: ivy.array([8, 6, 6]),
        b: ivy.array([[9., 0.7], [0.4, 0.]])
    }

    Example 7: Sorting a container x has two arrays, a and b,  in ascending order and returns the sorted container y.
    The order of the arrays in the container has changed as the first array a has smaller values compared to the second array b.

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


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
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
    """
    Return the indices of the inserted elements in a sorted array.

    Parameters
    ----------
    x (Union[ivy.Array, ivy.NativeArray]):
        Input array. If `sorter` is None, then it must be sorted in ascending order,
        otherwise `sorter` must be an array of indices that sort it.
    v (Union[ivy.Array, ivy.NativeArray]):
        specific elements to insert in array x1
    side (Literal["left", "right"], optional):
        The specific elements' index is at the 'left' side or
        'right' side in the sorted array x1. If the side is 'left', the
        index of the first suitable location located is given. If
        'right', return the last such index.
    ret_dtype (Union[ivy.Dtype, ivy.NativeDtype], optional):
        the data type for the return value, Default: ivy.int64,
        only integer data types is allowed.
    sorter (Optional[Union[ivy.Array, ivy.NativeArray, List[int]]], optional):
        optional array of integer indices that sort array x into ascending order,
        typically the result of argsort.
    out (Optional[ivy.Array], optional):
        optional output array, for writing the result to.

    Returns
    -------
    ret
         An array of insertion points.

    Examples
    --------
    With :class:`ivy.Array` input:

    Example 1: Searching for the insertion index of a single element in a sorted array

    >>> x = ivy.array([1, 2, 3])
    >>> v = ivy.array([2])
    >>> y  = ivy.searchsorted(x, v)
    >>> print(y)
    ivy.array([1])

    Example 2: Searching for the insertion index of a single element in a sorted array, with `side` parameter set to 'right

    >>> x = ivy.array([0, 1, 2, 3])
    >>> v = ivy.array([3])
    >>> y  = ivy.searchsorted(x, v, side='right')
    >>> print(y)
    ivy.array([4])

    Example 3: Searching for the insertion index of multiple elements in a sorted array

    >>> x = ivy.array([0, 1, 2, 3, 4, 5])
    >>> v = ivy.array([[3, 1], [10, 3], [-2, -1]])
    >>> y  = ivy.searchsorted(x, v)
    >>> print(y)
    ivy.array([[3, 1],
       [6, 3],
       [0, 0]])


    Example 4: Using out parameter to write result to a pre-allocated array

    >>> x = ivy.array([1, 2, 3])
    >>> v = ivy.array([2])
    >>> out = ivy.array([-1], dtype=ivy.int32)
    >>> y = ivy.searchsorted(x, v, out=out)
    >>> print(y)
    ivy.array([1])
    >>> print(out)
    ivy.array([1])
    """
    return ivy.current_backend(x, v).searchsorted(
        x,
        v,
        side=side,
        sorter=sorter,
        out=out,
        ret_dtype=ret_dtype,
    )
