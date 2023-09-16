# global
import numpy as np
from typing import Optional, Literal, Union, List

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def argsort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x: ndarray, (required)
        The input NumPy array to be sorted
    axis: int, (optional, default=-1)
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending: bool, (optional, default=False)
        direction  The direction in which to sort the values
    stable: bool, (optional, default=True)
        sort stability. If ``True``,
        the returned indices must maintain the relative order of ``x`` values which
        compare as equal. If ``False``, the returned indices may or may not maintain the
        relative order of ``x`` values which compare as equal (i.e., the relative order
        of ``x`` values which compare as equal is implementation-dependent).
        Default: ``True``.
    out: Optional[np.ndarray], (optional, default=None)
        optional output array, for writing the result to. It must have the same shape
        as ``x``.


    Returns
    -------
    ret
        Returns a new NumPy array containing the indices that would sort
        the elements of the input array x based on the specified sorting criteria.


    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = np.ndarray([3,1,2])
    >>> y = numpy.argsort(x)
    >>> print(y)
    np.ndarray([1,2,0])

    >>> x = np.ndarray([4,3,8])
    >>> y = numpy.argsort(x, descending=True)
    >>> print(y)
    np.ndarray([2,0,1])

    >>> x = np.ndarray([[1.5, 3.2], [2.3, 2.3]])
    >>> numpy.argsort(x, axis=0, descending=True, stable=False, out=x)
    >>> print(x)
    np.ndarray([[1, 0], [0, 1]])

    >>> x = np.ndarray([[[1,3], [3,2]], [[2,4], [2,0]]])
    >>> y = numpy.argsort(x, axis=1, descending=False, stable=True)
    >>> print(y)
    np.ndarray([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    """
    kind = "stable" if stable else "quicksort"
    return (
        np.argsort(-x, axis=axis, kind=kind)
        if descending
        else np.argsort(x, axis=axis, kind=kind)
    )


def sort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    x: ndarray
       The input NumPy array to be sorted
    axis: int, (optional, default=-1)
       axis along which to sort. If set to ``-1``, the function must sort along the
       last axis. Default: ``-1``.
    descending: bool, (optional, default=False)
       direction  The direction in which to sort the values
    stable: bool, (optional, default=True)
       sort stability. If ``True``,
       the returned indices must maintain the relative order of ``x`` values which
       compare as equal. If ``False``, the returned indices may or may not maintain the
       relative order of ``x`` values which compare as equal (i.e., the relative order
       of ``x`` values which compare as equal is implementation-dependent).
       Default: ``True``.
    out: Optional[np.ndarray], (optional, default=None)
       optional output array, for writing the result to. It must have the same shape
       as ``x``.


    Returns
    -------
    ret
        Returns a NumPy array containing the sorted elements from the input array x.


    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = np.ndarray([7, 8, 6])
    >>> y = numpy.sort(x)
    >>> print(y)
    np.ndarray([6, 7, 8])

    >>> x = np.ndarray([[[8.9,0], [19,5]],[[6,0.3], [19,0.5]]])
    >>> y = numpy.sort(x, axis=1, descending=True, stable=False)
    >>> print(y)
    np.ndarray([[[19. ,  5. ],[ 8.9,  0. ]],[[19. ,  0.5],[ 6. ,  0.3]]])

    >>> x = np.ndarray([1.5, 3.2, 0.7, 2.5])
    >>> y = numpy.zeros(5)
    >>> numpy.sort(x, descending=True, stable=False, out=y)
    >>> print(y)
    np.ndarray([3.2, 2.5, 1.5, 0.7])

    >>> x = np.ndarray([[1.1, 2.2, 3.3],[-4.4, -5.5, -6.6]])
    >>> numpy.sort(x, out=x)
    >>> print(x)
    np.ndarray([[ 1.1,  2.2,  3.3],
        [-6.6, -5.5, -4.4]])
    """
    kind = "stable" if stable else "quicksort"
    ret = np.asarray(np.sort(x, axis=axis, kind=kind))
    if descending:
        ret = np.asarray((np.flip(ret, axis)))
    return ret


# msort
@with_unsupported_dtypes({"1.25.2 and below": ("complex",)}, backend_version)
def msort(
    a: Union[np.ndarray, list, tuple], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return a copy of an array sorted along the first axis.

    Parameters
    ----------
    a: Union[np.ndarray, list, tuple], (required)
        The input array-like object (NumPy array, list, or tuple) that you want to sort.
        array-like input.
    out:  Optional[np.ndarray], (optional, default=None)
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns a new NumPy array containing the elements
        of the input a sorted in ascending order.


    Examples
    --------
    >>> a = np.ndarray([[8, 9, 6],[6, 2, 6]])
    >>> numpy.msort(a)
    np.ndarray(
        [[6, 2, 6],
         [8, 9, 6]]
        )
    """
    return np.msort(a)


msort.support_native_out = False


def searchsorted(
    x: np.ndarray,
    v: np.ndarray,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Union[np.ndarray, List[int]]] = None,
    ret_dtype: np.dtype = np.int64,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return the indices of the inserted elements in a sorted array.

    Parameters
    ----------
    x: np.ndarray, (required)
        The input NumPy array to be sorted
        Input array. If `sorter` is None, then it must be sorted in ascending order,
        otherwise `sorter` must be an array of indices that sort it.
    v: np.ndarray, (required)
        specific elements to insert in array x1
    side: Literal["left", "right"], (optional, default=left)
        The specific elements' index is at the 'left' side or
        'right' side in the sorted array x1. If the side is 'left', the
        index of the first suitable location located is given. If
        'right', return the last such index.
    sorter: Optional[Union[np.ndarray, List[int]]], (optional, default=None)
        optional array of integer indices that sort array x into ascending order,
        typically the result of argsort.
    ret_dtype: np.dtype, (optional, default=np.int64)
        the data type for the return value, Default: ivy.int64,
        only integer data types is allowed.
    out: Optional[np.ndarray], (optional, default=None)
        optional output array, for writing the result to.


    Returns
    -------
    ret
         Returns a NumPy array containing the indices where the values in v
         should be inserted into the sorted array x to maintain the sorted order.


    Examples
    --------
    With :class:`np.ndarray` input:

    >>> x = np.ndarray([1, 2, 3])
    >>> v = np.ndarray([2])
    >>> y  = numpy.searchsorted(x, v)
    >>> print(y)
    np.ndarray([1])

    >>> x = np.ndarray([0, 1, 2, 3])
    >>> v = np.ndarray([3])
    >>> y  = numpy.searchsorted(x, v, side='right')
    >>> print(y)
    np.ndarray([4])

    >>> x = np.ndarray([0, 1, 2, 3, 4, 5])
    >>> v = np.ndarray([[3, 1], [10, 3], [-2, -1]])
    >>> y  = numpy.searchsorted(x, v)
    >>> print(y)
    np.ndarray([[3, 1],
       [6, 3],
       [0, 0]])
    """
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    is_sorter_provided = sorter is not None
    if is_sorter_provided:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
    if x.ndim != 1:
        assert x.shape[:-1] == v.shape[:-1], RuntimeError(
            "the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )
        if is_sorter_provided:
            x = np.take_along_axis(x, sorter, axis=-1)
        original_shape = v.shape
        x = x.reshape(-1, x.shape[-1])
        v = v.reshape(-1, v.shape[-1])
        out_array = np.empty_like(v)
        for i in range(x.shape[0]):
            out_array[i] = np.searchsorted(x[i], v[i], side=side)
        ret = out_array.reshape(original_shape)
    else:
        ret = np.searchsorted(x, v, side=side, sorter=sorter)
    return ret.astype(ret_dtype)
