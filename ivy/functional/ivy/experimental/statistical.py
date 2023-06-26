from typing import Optional, Union, Tuple, Sequence
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    infer_dtype,
)
from ivy.utils.exceptions import handle_exceptions


# TODO: Make bins optional by offering an automatic bins creation like numpy.
#       Make density argument work in tensorflow
#       Bins as str is not defined (check Numpy implementation).
#       Permit multiple axis.
#       Modify documentation to match the above modifications.
@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def histogram(
    a: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    bins: Optional[Union[int, ivy.Array, ivy.NativeArray]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    density: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the histogram of the array ``a``.

    .. note::
        Given bins = [c0, ..., cK], defining intervals I0 = [c0, c1), I1 = [c1, c2),
        ..., I_{K-1} = [c_{K-1}, cK].

    Parameters
    ----------
    a
        input array.
    bins
        if ``bins`` is an int, it defines the number of equal-width bins in the given
        range.
        if ``bins`` is an array, it defines a monotonically increasing array of bin
        edges, including the rightmost edge, allowing for non-uniform bin widths.
    axis
        dimension along which maximum values must be computed. By default, the maximum
        value must be computed over the entire array. Default: ``None``.
    extend_lower_interval
        if True, extend the lowest interval I0 to (-inf, c1].
    extend_upper_interval
        ff True, extend the upper interval I_{K-1} to [c_{K-1}, +inf).
    dtype
        the output type.
    range
        the lower and upper range of the bins. The first element of the range must be
        less than or equal to the second.
    weights
        each value in ``a`` only contributes its associated weight towards the bin count
        (instead of 1). Must be of the same shape as a.
    density
        if True, the result is the value of the probability density function at the
        bin, normalized such that the integral over the range of bins is 1.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a tuple containing the values of the histogram and the bin edges.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 2])
    >>> y = ivy.array([0., 0.5, 1., 1.5, 2.])
    >>> z = ivy.histogram(x, bins=y)
    >>> print(z)
    (ivy.array([1, 0, 1, 1]), ivy.array([0. , 0.5, 1. , 1.5, 2. ]))

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [4.4, 5.5, .6]])
    >>> bins = 4
    >>> range = (0., 5.)
    >>> dtype = ivy.int32
    >>> y = ivy.histogram(x, bins=bins, range=range, dtype=dtype)
    >>> print(y)
    (ivy.array([0, 0, 0, 0]), ivy.array([0.   , 0.125, 0.25 , 0.375, 0.5  ]))

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
    >>> axis = 1
    >>> extend_lower_interval = True
    >>> extend_upper_interval = True
    >>> dtype = ivy.float32
    >>> weights = ivy.array([[1., 1., 1.], [1., 1., 1.]])
    >>> z = ivy.histogram(
    >>>                     x,
    >>>                     bins=y,
    >>>                     axis=axis,
    >>>                     extend_lower_interval=extend_lower_interval,
    >>>                     extend_upper_interval=extend_upper_interval,
    >>>                     dtype=dtype,
    >>>                     weights=weights)
    >>> print(z)
    (ivy.array([[0., 3.],
    [1., 0.],
    [1., 0.],
    [1., 0.],
    [0., 0.]]), ivy.array([0., 1., 2., 3., 4., 5.]))

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
    >>> dtype = ivy.int32
    >>> z = ivy.histogram(x, bins=y, dtype=dtype)
    >>> print(z.a)
    >>> print(z.b)
    (ivy.array([1, 1, 1, 0, 0]), ivy.array([0., 1., 2., 3., 4., 5.]))
    (ivy.array([0, 0, 0, 1, 2]), ivy.array([0., 1., 2., 3., 4., 5.]))
    """
    return ivy.current_backend(a).histogram(
        a,
        bins=bins,
        axis=axis,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        range=range,
        weights=weights,
        density=density,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def median(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the median along the specified axis.

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


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
def nanmean(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the mean of all non-NaN elements along the specified dimensions.

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


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
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
    """
    Compute the q-th quantile of the data along the specified axis.

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


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def corrcoef(
    x: ivy.Array,
    /,
    *,
    y: Optional[ivy.Array] = None,
    rowvar: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend().corrcoef(x, y=y, rowvar=rowvar, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def nanmedian(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    ivy.Array instance method variant of ivy.nanmedian. This method simply wraps the
    function, and so the docstring for ivy.nanmedian also applies to this method with
    minimal changes.

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

    This function is *nestable*, and therefore also accepts :code:'ivy.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[12.0, 10.0, 34.0], [45.0, 23.0, ivy.nan]])
    >>> ivy.nanmedian(x)
        ivy.array(23.)
    With a mix of :class:`ivy.Container` and :class:`ivy.Array` input:
    >>> x = ivy.Container(a=ivy.array([[10.0, ivy.nan, 4], [3, 2, 1]]),
            b=ivy.array([[12, 10, 34], [45, 23, ivy.nan]]))
    >>> ivy.nanmedian(x)
    {
        a: ivy.array(3.),
        b: ivy.array(23.)
    }
    """
    return ivy.current_backend().nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def bincount(
    x: ivy.Array,
    /,
    *,
    weights: Optional[ivy.Array] = None,
    minlength: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Count the number of occurrences of each value in an integer array.

    Parameters
    ----------
    self
        Input array.
    weights
        An optional input array.
    minlength
        A minimum number of bins for the output array.

    Returns
    -------
    ret
        The bincount of the array elements.

    Examples
    --------
    >>> a = ivy.Container([[10.0, ivy.nan, 4], [3, 2, 1]])
    >>> a.bincount(a)
        3.0
    >>> a.bincount(a, axis=0)
        array([6.5, 2. , 2.5])
    """
    return ivy.current_backend(x).bincount(
        x, weights=weights, minlength=minlength, out=out
    )


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def igamma(
    a: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    x: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Compute the regularized lower gamma function of ``a`` and ``x``.

    Parameters
    ----------
    self
        Input array.
    x
        An additional input array.
        `x` has the same type as `a`.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The lower incomplete gamma function of the array elements.

    Examples
    --------
    >>> a = ivy.array([2.5])
    >>> x = ivy.array([1.7, 1.2])
    >>> a.igamma(x)
        ivy.array([0.3614, 0.2085])
    """
    return ivy.current_backend().igamma(a, x=x, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def nanquantile(
    a: ivy.Array,
    q: Union[ivy.Array, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend(a).nanquantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=interpolation, out=out
    )
