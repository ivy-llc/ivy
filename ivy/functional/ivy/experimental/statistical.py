from typing import Optional, Union, Tuple, Sequence
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_nestable,
    infer_dtype,
    handle_device_shifting,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions


# TODO: Make bins optional by offering an automatic bins creation like numpy.
#       Make density argument work in tensorflow
#       Bins as str is not defined (check Numpy implementation).
#       Permit multiple axis.
#       Modify documentation to match the above modifications.
@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
    ivy.array([1., 0., 1., 1.])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [4.4, 5.5, .6]])
    >>> bins = 4
    >>> range = (0., 5.)
    >>> dtype = ivy.int32
    >>> y = ivy.histogram(x, bins=bins, range=range, dtype=dtype)
    >>> print(y)
    ivy.array([2, 1, 1, 1])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
    >>> axis = 1
    >>> extend_lower_interval = True
    >>> extend_upper_interval = True
    >>> dtype = ivy.float32
    >>> weights = ivy.array([[1., 1., 1.], [1., 1., 1.]])
    >>> z = ivy.histogram(
    ...                     x,
    ...                     bins=y,
    ...                     axis=axis,
    ...                     extend_lower_interval=extend_lower_interval,
    ...                     extend_upper_interval=extend_upper_interval,
    ...                     dtype=dtype,
    ...                     weights=weights)
    >>> print(z)
    ivy.array([[0., 3.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 0.]])

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.array([0., 1., 2., 3., 4., 5.])
    >>> dtype = ivy.int32
    >>> z = ivy.histogram(x, bins=y, dtype=dtype)
    >>> print(z)
    {
        a: ivy.array([1, 1, 1, 0, 0]),
        b: ivy.array([0, 0, 0, 1, 2])
    }
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device_shifting
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device_shifting
def nanprod(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the product of array elements over a given axis treating Not a Numbers
    (NaNs) as ones.

    Parameters
    ----------
    a
        Input array.
    axis
        Axis or axes along which the product is computed.
        The default is to compute the product of the flattened array.
    dtype
        The desired data type of returned array. Default is None.
    out
        optional output array, for writing the result to.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the original a.
    initial
        The starting value for this product.
    where
        Elements to include in the product

    Returns
    -------
    ret
        The product of array elements over a given axis treating
        Not a Numbers (NaNs) as ones

    Functional Examples
    -------------------
    >>> a = ivy.array([[1, ivy.nan], [3, 4]])
    >>> ivy.nanprod(a)
    12.0
    """
    return ivy.current_backend(a).nanprod(
        a,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        out=out,
        initial=initial,
        where=where,
    )


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
        {'nearest', 'linear', 'lower', 'higher', 'midpoint', 'nearest_jax'}.
        Default value: 'linear'.
        This specifies the interpolation method to use when the desired quantile lies
        between two data points i < j:
        - linear: i + (j - i) * fraction, where fraction is the fractional part of the
        index surrounded by i and j.
        - lower: i.
        - higher: j.
        - nearest: i or j, whichever is nearest.
        - midpoint: (i + j) / 2. linear and midpoint interpolation do not work with
        integer dtypes.
        - nearest_jax: provides jax-like computation for interpolation='nearest'.
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
@handle_backend_invalid
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


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def cov(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray] = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[ivy.Array] = None,
    aweights: Optional[ivy.Array] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
) -> ivy.Array:
    """
    Compute the covariance of matrix x1, or variables x1 and x2.

    Parameters
    ----------
    x1
        a 1D or 2D input array, with a numeric data type.
    x2
        optional second 1D or 2D input array, with a numeric data type.
        Must have the same shape as ``self``.
    rowVar
        optional variable where each row of input is interpreted as a variable
        (default = True). If set to False, each column is instead interpreted
        as a variable.
    bias
        optional variable for normalizing input (default = False) by (N - 1) where
        N is the number of given observations. If set to True, then normalization
        is instead by N. Can be overridden by keyword ``ddof``.
    ddof
        optional variable to override ``bias`` (default = None). ddof=1 will return
        the unbiased estimate, even with fweights and aweights given. ddof=0 will
        return the simple average.
    fweights
        optional 1D array of integer frequency weights; the number of times each
        observation vector should be repeated.
    aweights
        optional 1D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ddof=0 is specified, the array
        of weights can be used to assign probabilities to observation vectors.
    dtype
        optional variable to set data-type of the result. By default, data-type
        will have at least ``numpy.float64`` precision.
    out
        optional output array, for writing the result to. It must have a shape that
        the inputs broadcast to.

    Returns
    -------
    ret
        an array containing the covariance matrix of an input matrix, or the
        covariance matrix of two variables. The returned array must have a
        floating-point data type determined by Type Promotion Rules and must be
        a square matrix of shape (N, N), where N is the number of variables in the
        input(s).

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    extensions/generated/signatures.linalg.cov.html>`_
    in the standard.
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> y = x[0].cov(x[1])
    >>> print(y)
    ivy.array([[1., 1.],
           [1., 1.]])

    With :class:`ivy.Container` inputs:
    >>> x = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([1., 2., 3.]))
    >>> y = ivy.Container(a=ivy.array([3., 2., 1.]), b=ivy.array([3., 2., 1.]))
    >>> z = ivy.Container.static_cov(x, y)
    >>> print(z)
    {
        a: ivy.array([[1., -1.],
                      [-1., 1.]]),
        b: ivy.array([[1., -1.],
                      [-1., 1.]])
    }

    With a combination of :class:`ivy.Array` and :class:`ivy.Container` inputs:
    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.Container(a=ivy.array([3. ,2. ,1.]), b=ivy.array([-1., -2., -3.]))
    >>> z = ivy.cov(x, y)
    >>> print(z)
    {
        a: ivy.array([[1., -1.],
                      [-1., 1.]]),
        b: ivy.array([[1., -1.],
                      [-1., 1.]])
    }

    With :class:`ivy.Array` input and rowVar flag set to False (True by default):
    >>> x = ivy.array([[1,2,3],
    ...                [4,5,6]])
    >>> y = x[0].cov(x[1], rowVar=False)
    >>> print(y)
    ivy.array([[1., 1.],
           [1., 1.]])

    With :class:`ivy.Array` input and bias flag set to True (False by default):
    >>> x = ivy.array([[1,2,3],
    ...                [4,5,6]])
    >>> y = x[0].cov(x[1], bias=True)
    >>> print(y)
    ivy.array([[0.66666667, 0.66666667],
           [0.66666667, 0.66666667]])

    With :class:`ivy.Array` input with both fweights and aweights given:
    >>> x = ivy.array([[1,2,3],
    ...                [4,5,6]])
    >>> fw = ivy.array([1,2,3])
    >>> aw = ivy.array([ 1.2, 2.3, 3.4 ])
    >>> y = x[0].cov(x[1], fweights=fw, aweights=aw)
    >>> print(y)
    ivy.array([[0.48447205, 0.48447205],
           [0.48447205, 0.48447205]])
    """
    return ivy.current_backend(x1).cov(
        x1,
        x2,
        rowVar=rowVar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        dtype=dtype,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def cummax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a tuple containing the cumulative maximum of elements of input along the
    given axis and index location of each maximum value found along the given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which the cumulative maximum is computed. Default is ``0``.
    exclusive
        Whether to perform cummax exclusively. Default is ``False``.
    reverse
        Whether to perform the cummax from last to first element in the selected
        axis. Default is ``False`` (from first to last element)
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Array which holds the result of applying cummax at each
        original array elements along the specified axis.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-86, -19, 41, 88, -5, 80, 32, 87, -90, -12])
    >>> y = ivy.cummax(x, exclusive=False, reverse=False)
    >>> print(y)
    (ivy.array([-86, -19,  41,  88,  88,  88,  88,  88,  88,  88]),
    ivy.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 3]))

    >>> x = ivy.array([ 14,  15,  49, -24, -39])
    >>> y = ivy.cummax(x, axis=0, exclusive=False, reverse=False)
    >>> print(y)
    (ivy.array([14, 15, 49, 49, 49]), ivy.array([0, 1, 2, 2, 2]))

    >>> x = ivy.array([[ 63,  43, -16,  -4],[ 21,  82,  59,  33]])
    >>> ivy.cummax(x, axis=0, reverse=False, dtype='int64', out=x)
    >>> print(x)
    ivy.array([[0, 0, 0, 0],
           [0, 1, 1, 1]])

    >>> x = ivy.array([[-36,  83, -81],
    ...                [ 23,  29,  63],
    ...                [-83,  85,   2],
    ...                [ 31,  25, -86],
    ...                [-10, -52,   0],
    ...                [ 22,  38,  55],
    ...                [ 33,  54, -16]])
    >>> y = ivy.cummax(x, axis=1, exclusive=True, reverse=False)
    >>> print(y)
    (ivy.array([[ 0,  0, 83],
           [ 0, 23, 29],
           [ 0,  0, 85],
           [ 0, 31, 31],
           [ 0,  0,  0],
           [ 0, 22, 38],
           [ 0, 33, 54]]), ivy.array([[0, 0, 2],
           [0, 1, 2],
           [0, 0, 2],
           [0, 1, 1],
           [0, 0, 0],
           [0, 1, 2],
           [0, 1, 2]]))

    >>> x = ivy.array([73, 15, 47])
    >>> y = ivy.cummax(x, axis=0, reverse=True, exclusive=True)
    >>> print(y)
    (ivy.array([47, 47,  0]), ivy.array([0, 0, 0]))

    >>> x = ivy.array([-47, -14, -67, 15, -23, -45])
    >>> y = ivy.cummax(x, axis=0, reverse=True, exclusive=False)
    >>> print(y)
    (ivy.array([ 15,  15,  15,  15, -23, -45]), ivy.array([2, 2, 2, 2, 1, 0]))
    """
    return ivy.current_backend(x).cummax(
        x, axis=axis, exclusive=exclusive, reverse=reverse, dtype=dtype, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def cummin(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the cumulative minimum of the elements along a given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which the cumulative minimum is computed. Default is ``0``.
    reverse
        Whether to perform the cummin from last to first element in the selected
        axis. Default is ``False`` (from first to last element)
    dtype
        Data type of the returned array. Default is ``None``.
        If None, if the default data type corresponding to the data type “kind”
        (integer or floating-point) of x has a smaller range of values than the
        data type of x (e.g., x has data type int64 and the default data type
        is int32, or x has data type uint64 and the default data type is int64),
        the returned array must have the same data type as x.
        If x has a floating-point data type, the returned array must have the
        default floating-point data type.
        If x has a signed integer data type (e.g., int16), the returned array
        must have the default integer data type.
        If x has an unsigned integer data type (e.g., uint16), the returned
        array must have an unsigned integer data type having the same number of
        bits as the default integer data type (e.g., if the default integer data
        type is int32, the returned array must have a uint32 data type).
        If the data type (either specified or resolved) differs from the data type
        of x, the input array should be cast to the specified data type before
        computing the product.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Array which holds the result of applying cummin at each
        original array elements along the specified axis.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 5, 2, 0])
    >>> y = ivy.cummin(x)
    >>> print(y)
    ivy.array([1, 1, 1, 0])
    >>> x = ivy.array([[6, 4, 2],
    ...                [1, 3, 0]])
    >>> y = ivy.zeros((2,3))
    >>> ivy.cummin(x, axis=0, reverse=True, out=y)
    >>> print(y)
    ivy.array([[1., 3., 0.],
           [1., 3., 0.]])

    >>> x = ivy.array([[2, 4, 5],
    ...                [3, 6, 5],
    ...                [1, 3, 10]])
    >>> ivy.cummin(x,axis=1,reverse=True, dtype='int64', out=x)
    >>> print(x)
    ivy.array([[ 2,  4,  5],
           [ 3,  5,  5],
           [ 1,  3, 10]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[1, 3, 5]]),
    ...                   b=ivy.array([[3, 5, 7]]))
    >>> y = ivy.cummin(x, axis= 0)
    >>> print(y)
    {
        a: ivy.array([[1, 3, 5]]),
        b: ivy.array([[3, 5, 7]])
    }

    >>> x = ivy.Container(a=ivy.array([[1, 3, 4]]),
    ...                   b=ivy.array([[3, 5, 8],
    ...                                [5, 6, 5]]),
    ...                   c=ivy.array([[2, 4, 1],
    ...                                [3, 6, 9],
    ...                                [0, 2, 3]]))
    >>> y = ivy.Container(a = ivy.zeros((1, 3)),
    ...                   b = ivy.zeros((2, 3)),
    ...                   c = ivy.zeros((3,3)))
    >>> ivy.cummin(x,axis=1,reverse=True, out=y)
    >>> print(y)
    {
        a: ivy.array([[1., 3., 4.]]),
        b: ivy.array([[3., 5., 8.],
                      [5., 5., 5.]]),
        c: ivy.array([[1., 1., 1.],
                      [3., 6., 9.],
                      [0., 2., 3.]])
    }

    >>> x = ivy.Container(a=ivy.array([[0],[5]]),
    ...                   b=ivy.array([[6, 8, 7],
    ...                                [4, 2, 3]]),
    ...                   c=ivy.array([[1, 2],
    ...                                [3, 4],
    ...                                [6, 4]]))
    >>> ivy.cummin(x,axis=0,out=x)
    >>> print(x)
    {
        a: ivy.array([[0],
                      [0]]),
        b: ivy.array([[6, 8, 7],
                      [4, 2, 3]]),
        c: ivy.array([[1, 2],
                      [1, 2],
                      [1, 2]])
    }
    """
    return ivy.current_backend(x).cummin(
        x, axis=axis, exclusive=exclusive, reverse=reverse, dtype=dtype, out=out
    )
