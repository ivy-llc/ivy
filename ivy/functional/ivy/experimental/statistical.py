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
@infer_dtype
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
) -> Tuple:
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
        Tuple with arrays of type int32 that have the same shape as the indices array.

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
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
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
    rowvar: Optional[bool] = True,
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
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
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
    minlength: Optional[int] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend(x).bincount(
        x, weights=weights, minlength=minlength, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def percentile(
    a: Union[ivy.Array, ivy.NativeArray],
    q: Union[Sequence[float], float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculates the q-th percentile of the array ``a``.

    .. note::
       When the number of elements over which to compute the q-th percentile is zero,
       some specification-compliant libraries may choose to raise an error, 
       (e.g., if ``a`` is a ``NaN`` input array, return ``NaN``),

    **Special Cases**

    Let ``x_i`` is the element from ``a`` or just ``a`` itself.

    -   If ``x_i`` is ``NaN``, the q-th percentile is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    a
        input array. Should have a floating-point data type.
    q
        Percentile or sequence of percentiles to compute, which must be between 0 and 
        100 inclusive.
    axis
        axis or axes along which q-th percentile must be computed. By default, the 
        percentile must be computed over the entire array. Default: ``None``.
    keepdims
        bool, if ``True``, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible with
        the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.
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
        scalar or array, if the q-th percentile was computed over the entire array, a
        zero-dimensional array containing the q-th percentile; otherwise, a
        non-zero-dimensional array containing the q-th percentile. The returned
        array must have the same data type as ``a``.
        .. note::
           While this specification recommends that this function only accept input
           arrays having a floating-point data type, specification-compliant array
           libraries may choose to accept input arrays having an integer data type.
           While mixed data type promotion is implementation-defined, if the input
           array ``a`` has an integer data type, the returned array must have the
           default floating-point data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/
    signatures.statistical_functions.mean.html>`_ in the standard.

    Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([3., 4., 5.])
    >>> y = ivy.percentile(x,75)
    >>> print(y)
    ivy.array(4.5)

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.array(0.)
    >>> ivy.percentile(x,75, out=y)
    >>> print(y)
    ivy.array(1.5)

    >>> x = ivy.array([[-1., -2., -3., 0., -1.], [1., 2., 3., 0., 1.]])
    >>> y = ivy.array([0., 0.])
    >>> ivy.percentile(x,75, axis=1, out=y)
    >>> print(y)
    ivy.array([-1.,  2.])

      >>> x = ivy.array([[-1., -2., -3., 0., -1.], [1., 2., 3., 0., 1.]])
    >>> y = ivy.array([0., 0.])
    >>> ivy.percentile(x,[75,90], axis=1, out=y)
    >>> print(y)
    ivy.array([[-1.        ,  2.        ],
       [-0.40000001,  2.5999999 ]])

    With :class:`ivy.Array` input:

    >>> x = ivy.array([3., 4., 5.])
    >>> y = ivy.percentile(x,75)
    >>> print(y)
    ivy.array(4.5)

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.array(0.)
    >>> ivy.percentile(x,75, out=y)
    >>> print(y)
    ivy.array(1.5)

    >>> x = ivy.array([[-1., -2., -3., 0., -1.], [1., 2., 3., 0., 1.]])
    >>> y = ivy.array([0., 0.])
    >>> ivy.percentile(x,75, axis=1, out=y)
    >>> print(y)
    ivy.array([-1.,  2.])

    >>> x = ivy.array([[-1., -2., -3., 0., -1.], [1., 2., 3., 0., 1.]])
    >>> y = ivy.array([0., 0.])
    >>> ivy.percentile(x,[75,90], axis=1, out=y)
    >>> print(y)
    ivy.array([[-1.        ,  2.        ],
       [-0.40000001,  2.5999999 ]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1., 0., 1.]), b=ivy.array([1.1, 0.2, 1.4]))
    >>> y = ivy.percentile(x,75)
    >>> print(y)
    {
        a: ivy.array(0.5),
        b: ivy.array(1.25)
    }


    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.], [3., 4., 5.]]),
    ...                   b=ivy.array([[3., 4., 5.], [6., 7., 8.]]))
    >>> ivy.percentile(x,75, axis=0, out=x)
    >>> print(x)
    {
        a: ivy.array([2.25, 3.25, 4.25]),
        b: ivy.array([5.25, 6.25, 7.25])
    }

    """
    return ivy.current_backend(a).percentile(a, q, axis=axis, keepdims=keepdims, 
                                             interpolation=interpolation, out=out)
