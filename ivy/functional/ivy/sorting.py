# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable
)


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def argsort(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the indices that sort an array ``x`` along a specified axis.

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

    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/
        signatures.elementwise_functions.tan.html>`_
    in the standard. The descriptions above assume an array input for simplicity, but
    the method also accepts :code:`ivy.Container` instances in place of
    :code:`ivy.Array` or :code:`ivy.NativeArray` instances, as shown in the type hints
    and also the examples below.

    Functional Examples
    -------------------

    With: code:`ivy.Array` input:

    >>> x = ivy.array([3,1,2])
    >>> y = ivy.argsort(x)
    >>> print(y)
    ivy.array([1,2,0])

    >>> x = ivy.array([[1.5, 3.2], [2.3, 2.3]])
    >>> ivy.argsort(x, 0, True, False, out=x)
    >>> print(x)
    ivy.array([[1, 0], [0, 1]])

    >>> x = ivy.array([[[1,3], [3,2]], [[2,4], [2,0]]])
    >>> y = ivy.argsort(x, 1, False, True)
    >>> print(y)
    ivy.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([5,1,3]), b=ivy.array([[0, 3], [3, 2]]))
    >>> y = ivy.argsort(x)
    >>> print(y)
    {
        a: ivy.array([1, 2, 0]),
        b: ivy.array([[0, 1], [1, 0]])
    }
    """
    return ivy.current_backend(x).argsort(x, axis, descending, stable, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Returns a sorted copy of an array.

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
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the same dtype and shape as `values`, with the elements sorted
        along the given `axis`.

    """
    return ivy.current_backend(x).sort(x, axis, descending, stable, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def searchsorted(
    x: Union[ivy.Array, ivy.NativeArray],
    v: Union[ivy.Array, ivy.NativeArray],
    side: str = "left",
    sorter: bool = None,
    ret_dtype: int = ivy.int64,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the indices of the inserted elements in a sorted array.
    
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
    return ivy.current_backend(x, v).searchsorted(x, v, side, sorter, ret_dtype, out)
