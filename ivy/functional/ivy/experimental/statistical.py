from typing import Optional, Union, Tuple, Sequence
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    infer_dtype,
)
from ivy.utils.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def median(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
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


@infer_dtype
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nanmean(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def corrcoef(
    x: ivy.Array,
    /,
    *,
    y: Optional[ivy.Array] = None,
    rowvar: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend().corrcoef(x, y=y, rowvar=rowvar, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nanmedian(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """ivy.Array instance method variant of ivy.nanmedian. This method simply
    wraps the function, and so the docstring for ivy.nanmedian also applies to
    this method with minimal changes.

    Parameters
    ----------
    self
        Input array.
    axis
        Axis or axes along which the means are computed.
        The default is to compute the mean of the flattened array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the original a. If the value is anything but the default,
        then keepdims will be passed through to the mean or sum methods of
        sub-classes of ndarray. If the sub-classes methods does not implement
        keepdims any exceptions will be raised.
    overwrite_input
        If True, then allow use of memory of input array a for calculations.
        The input array will be modified by the call to median. This will
        save memory when you do not need to preserve the contents of the input array.
        Treat the input as undefined, but it will probably be fully or partially sorted.
        Default is False. If overwrite_input is True and a is not already an ndarray,
        an error will be raised.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        A new array holding the result. If the input contains integers

    Examples
    --------
    >>> a = ivy.Array([[10.0, ivy.nan, 4], [3, 2, 1]])
    >>> a.nanmedian(a)
        3.0
    >>> a.nanmedian(a, axis=0)
        array([6.5, 2. , 2.5])
    """
    return ivy.current_backend().nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def bincount(
    x: ivy.Array,
    /,
    *,
    weights: Optional[ivy.Array] = None,
    minlength: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend(x).bincount(
        x, weights=weights, minlength=minlength, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def average(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the mean of all elements along the specified dimensions.
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
        The mean of the array elements.

    Examples
    --------
    >>> a = ivy.array([0.2294, -0.5481, 1.3288])
    >>> ivy.average(a)
    0.3367
    >>> a = ivy.array([[-0.3841,  0.6320,  0.4254, -0.7384],
    >>>                [-0.9644,  1.0131, -0.6549, -1.4279],
    >>>                [-0.2951, -1.3350, -0.7694,  0.5600],
    >>>                [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> ivy.average(a,1)
    ivy.array([-0.0163, -0.5085, -0.4599,  0.1807])
    """
    return ivy.current_backend(a).average(
        a, axis=axis, keepdims=keepdims, dtype=dtype, out=out
    )
