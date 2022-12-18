from typing import Optional, Union, Tuple, Sequence
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def median(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the median along the specified axis.

    Parameters
    ----------
    input
        Input array.
    axis
        Axis or axes along which the medians are computed. The default is to compute
        the median along a flattened version of the array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The median of the array elements.

    Functional Examples
    -------------------
    >>> a = ivy.array([[10, 7, 4], [3, 2, 1]])
    >>> ivy.median(a)
    3.5
    >>> ivy.median(a, axis=0)
    ivy.array([6.5, 4.5, 2.5])
    """
    return ivy.current_backend().median(input, axis=axis, keepdims=keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nanmean(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the mean of all non-NaN elements along the specified dimensions.

    Parameters
    ----------
    a
        Input array.
    axis
        Axis or axes along which the means are computed.
        The default is to compute the mean of the flattened array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the original a. If the value is anything but the default,
        then keepdims will be passed through to the mean or sum methods of sub-classes
        of ndarray. If the sub-classes methods does not implement keepdims any
        exceptions will be raised.
    dtype
        The desired data type of returned tensor. Default is None.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The nanmean of the array elements.

    Functional Examples
    -------------------
    >>> a = ivy.array([[1, ivy.nan], [3, 4]])
    >>> ivy.nanmean(a)
    2.6666666666666665
    >>> ivy.nanmean(a, axis=0)
    ivy.array([2.,  4.])
    """
    return ivy.current_backend(a).nanmean(
        a, axis=axis, keepdims=keepdims, dtype=dtype, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def unravel_index(
    indices: ivy.Array,
    shape: Tuple[int],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Converts a flat index or array of flat indices
    into a tuple of coordinate arrays.

    Parameters
    ----------
    indices
        Input array.
    shape
        The shape of the array to use for unraveling indices.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Tuple with arrays that have the same shape as the indices array.

    Functional Examples
    -------------------
    >>> indices = ivy.array([22, 41, 37])
    >>> ivy.unravel_index(indices, (7,6))
    (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
    """
    return ivy.current_backend(indices).unravel_index(indices, shape, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def quantile(
    a: ivy.Array,
    q: Union[ivy.Array, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a
        Input array.
    q
        Quantile or sequence of quantiles to compute, which must be
        between 0 and 1 inclusive.
    axis
        Axis or axes along which the quantiles are computed. The default
        is to compute the quantile(s) along a flattened version of the array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the original array a.
    interpolation
        {'nearest', 'linear', 'lower', 'higher', 'midpoint'}. Default value: 'linear'.
        This specifies the interpolation method to use when the desired quantile lies
        between two data points i < j:
        - linear: i + (j - i) * fraction, where fraction is the fractional part of the
        index surrounded by i and j.
        - lower: i.
        - higher: j.
        - nearest: i or j, whichever is nearest.
        - midpoint: (i + j) / 2. linear and midpoint interpolation do not work with
        integer dtypes.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        A (rank(q) + N - len(axis)) dimensional array of same dtype as a, or, if axis
        is None, a rank(q) array. The first rank(q) dimensions index quantiles for
        different values of q.

    Examples
    --------
    >>> a = ivy.array([[10., 7., 4.], [3., 2., 1.]])
    >>> q = ivy.array(0.5)
    >>> ivy.quantile(a, q)
    ivy.array(3.5)

    >>> a = ivy.array([[10., 7., 4.], [3., 2., 1.]])
    >>> q = 0.5
    >>> ivy.quantile(a, q)
    ivy.array(3.5)

    >>> ivy.quantile(a, q, axis=0)
    ivy.array([6.5, 4.5, 2.5])

    >>> ivy.quantile(a, q, axis=1)
    ivy.array([7.,  2.])

    >>> ivy.quantile(a, q, axis=1, keepdims=True)
    ivy.array([[7.],[2.]])

    >>> a = ivy.array([1., 2., 3., 4.])
    >>> q = ivy.array([0.3, 0.7])
    >>> ivy.quantile(a, q, interpolation='lower')
    ivy.array([1., 3.])
    """
    return ivy.current_backend(a).quantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=interpolation, out=out
    )
