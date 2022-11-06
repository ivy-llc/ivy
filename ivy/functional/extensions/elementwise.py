# local
from typing import Optional, Union
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
)
from ivy.exceptions import handle_exceptions


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def sinc(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculates an implementation-dependent approximation of the principal value of
    the normalized sinc function, having domain ``(-infinity, +infinity)`` and
    codomain ``[-0.217234, 1]``, for each element ``x_i`` of the input array ``x``.
    Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If x_i is NaN, the result is NaN.
    - If ``x_i`` is ``0``, the result is ``1``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the normalized sinc function of each element in x.
        The returned array must have a floating-point data type determined
        by :ref:`type-promotion`.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
    >>> y = x.sinc()
    >>> print(y)
    ivy.array([0.637,-0.212,0.127,-0.0909])

    >>> x = ivy.array([1.5, 0.5, -1.5])
    >>> y = ivy.zeros(3)
    >>> ivy.sinc(x, out=y)
    >>> print(y)
    ivy.array([-0.212,0.637,-0.212])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
    >>> y = ivy.sinc(x)
    >>> print(y)
    ivy.array([0.637,-0.212,0.127,-0.0909])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
    ...                   b=ivy.array([3.5, 4.5, 5.5]))
    >>> y = x.sinc()
    >>> print(y)
    {
        a: ivy.array([0.637,-0.212,0.127]),
        b: ivy.array([-0.0909,0.0707,-0.0579])
    }
    """
    return ivy.current_backend(x).sinc(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def lcm(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the element-wise least common multiple (LCM) of x1 and x2.

    Parameters
    ----------
    x1
        first input array.
    x2
        second input array
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        an array that includes the element-wise least common multiples of x1 and x2

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1=ivy.array([2, 3, 4])
    >>> x2=ivy.array([5, 8, 15])
    >>> x1.lcm(x1, x2)
    ivy.array([10, 21, 60])
    """
    return ivy.current_backend().lcm(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def fmod(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Computes the element-wise remainder of divisions of two arrays.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with element-wise remainder of divisions.

    Examples
    --------
    >>> x1 = ivy.array([2, 3, 4])
    >>> x2 = ivy.array([1, 5, 2])
    >>> ivy.fmod(x1, x2)
    ivy.array([ 0,  3,  0])

    >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
    >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
    >>> ivy.fmod(x1, x2)
    ivy.array([ nan,  nan,  nan])
    """
    return ivy.current_backend().fmod(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def fmax(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Computes the element-wise maximums of two arrays. Differs from ivy.maximum
    in the case where one of the elements is NaN. ivy.maximum returns the NaN element
    while ivy.fmax returns the non-NaN element.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with element-wise maximums.

    Examples
    --------
    >>> x1 = ivy.array([2, 3, 4])
    >>> x2 = ivy.array([1, 5, 2])
    >>> ivy.fmax(x1, x2)
    ivy.array([ 2.,  5.,  4.])

    >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
    >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
    >>> ivy.fmax(x1, x2)
    ivy.array([ 0,  0,  nan])
    """
    return ivy.current_backend().fmax(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def trapz(
    y: ivy.Array,
    /,
    *,
    x: Optional[ivy.Array] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Integrate along the given axis using the composite trapezoidal rule.
    If x is provided, the integration happens in sequence along its elements
    - they are not sorted..

    Parameters
    ----------
    y
        The array that should be integrated.
    x
        The sample points corresponding to the input array values.
        If x is None, the sample points are assumed to be evenly spaced
        dx apart. The default is None.
    dx
        The spacing between sample points when x is None. The default is 1.
    axis
        The axis along which to integrate.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Definite integral of n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If the input array is a
        1-dimensional array, then the result is a float. If n is greater
        than 1, then the result is an n-1 dimensional array.

    Examples
    --------
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3])
    4.0
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3], x=[4, 6, 8])
    8.0
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.trapz([1,2,3], dx=2)
    8.0
    """
    return ivy.current_backend().trapz(y, x=x, dx=dx, axis=axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def float_power(
    x1: Union[ivy.Array, float, list, tuple],
    x2: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Raise each base in x1 to the positionally-corresponding power in x2.
    x1 and x2 must be broadcastable to the same shape.
    This differs from the power function in that integers, float16, and float32
    are promoted to floats with a minimum precision of float64 so that the result
    is always inexact.

    Parameters
    ----------
    x1
        Array-like with elements to raise in power.
    x2
        Array-like of exponents. If x1.shape != x2.shape,
        they must be broadcastable to a common shape
        (which becomes the shape of the output).
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The bases in x1 raised to the exponents in x2.
        This is a scalar if both x1 and x2 are scalars

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3, 4, 5])
    >>> ivy.float_power(x1, 3)
    ivy.array([1.,    8.,   27.,   64.,  125.])
    >>> x1 = ivy.array([1, 2, 3, 4, 5])
    >>> x2 = ivy.array([2, 3, 3, 2, 1])
    >>> ivy.float_power(x1, x2)
    ivy.array([1.,   8.,  27.,  16.,   5.])
    """
    return ivy.current_backend().float_power(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def exp2(
    x: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate 2**p for all p in the input array.

    Parameters
    ----------
    x
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise 2 to the power x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> ivy.exp2(x)
    ivy.array([2.,    4.,   8.])
    >>> x = [5, 6, 7]
    >>> ivy.exp2(x)
    ivy.array([32.,   64.,  128.])
    """
    return ivy.current_backend().exp2(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nansum(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[tuple, int]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis or axes along which the sum is computed.
        The default is to compute the sum of the flattened array.
    dtype
        The type of the returned array and of the accumulator in
        which the elements are summed. By default, the dtype of input is used.
    keepdims
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.
    out
        Alternate output array in which to place the result.
        The default is None.

    Returns
    -------
    ret
        A new array holding the result is returned unless out is specified,
        in which it is returned.

    Examples
    --------
    >>> a = ivy.array([[ 2.1,  3.4,  ivy.nan], [ivy.nan, 2.4, 2.1]])
    >>> ivy.nansum(a)
    10.0
    >>> ivy.nansum(a, axis=0)
    ivy.array([2.1, 5.8, 2.1])
    >>> ivy.nansum(a, axis=1)
    ivy.array([5.5, 4.5])
    """
    return ivy.current_backend().nansum(
        x, axis=axis, dtype=dtype, keepdims=keepdims, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def gcd(
    x1: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the greatest common divisor of |x1| and |x2|.

    Parameters
    ----------
    x1
        First array-like input.
    x2
        Second array-input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise gcd of |x1| and |x2|.

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> ivy.gcd(x1, x2)
    ivy.array([1.,    1.,   3.])
    >>> x1 = ivy.array([1, 2, 3])
    >>> ivy.gcd(x1, 10)
    ivy.array([1.,   2.,  1.])
    """
    return ivy.current_backend().gcd(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def isclose(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Returns a boolean array where two arrays are element-wise equal
    within a tolerance.
    The tolerance values are positive, typically very small numbers.
    The relative difference (rtol * abs(b)) and the absolute difference
    atol are added together to compare against the absolute difference
    between a and b.
    The default atol is not appropriate for comparing numbers that are
    much smaller than one

    Parameters
    ----------
    a
        First input array.
    b
        Second input array.
    rtol
        The relative tolerance parameter.
    atol
        The absolute tolerance parameter.
    equal_nan
        Whether to compare NaN's as equal. If True, NaN's in a will be
        considered equal to NaN's in b in the output array.
    out
        Alternate output array in which to place the result.
        The default is None.

    Returns
    -------
    ret
        Returns a boolean array of where a and b are equal within the given
        tolerance. If both a and b are scalars, returns a single boolean value.

    Examples
    --------
    >>> ivy.isclose([1e10,1e-7], [1.00001e10,1e-8])
    ivy.array([True, False])
    >>> ivy.isclose([1.0, ivy.nan], [1.0, ivy.nan], equal_nan=True)
    ivy.array([True, True])
    >>> ivy.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
    ivy.array([False, False])
    >>> ivy.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], rtol=0.005, atol=0.0)
    ivy.array([False, True])
    """
    return ivy.current_backend().isclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def isposinf(
    x: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns a boolean array with values True where 
        the corresponding element of the input is positive
        infinity and values False where the element of the
        input is not positive infinity.

    Examples
    --------
    >>> x = ivy.array([1, 2, ivy.inf])
    >>> ivy.isposinf(x)
    ivy.array([False, False,  True])
    >>> x = [5, -ivy.inf, ivy.inf]
    >>> ivy.isposinf(x)
    ivy.array([False, False,  True])
    """
    return ivy.current_backend().isposinf(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def isneginf(
    x: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns a boolean array with values True where 
        the corresponding element of the input is negative
        infinity and values False where the element of the
        input is not negative infinity.

    Examples
    --------
    >>> x = ivy.array([1, 2, -ivy.inf])
    >>> ivy.isneginf(x)
    ivy.array([False, False,  True])
    >>> x = [5, -ivy.inf, ivy.inf]
    >>> ivy.isneginf(x)
    ivy.array([False, True,  False])
    """
    return ivy.current_backend().isneginf(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def nan_to_num(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Replace NaN with zero and infinity with large finite numbers
    (default behaviour) or with the numbers defined by the user using
    the nan, posinf and/or neginf keywords.

    Parameters
    ----------
    x
        Array input.
    copy
        Whether to create a copy of x (True) or to replace values in-place (False).
        The in-place operation only occurs if casting to an array does not require
        a copy. Default is True.
    nan
        Value to be used to fill NaN values. If no value is passed then NaN values
        will be replaced with 0.0.
    posinf
        Value to be used to fill positive infinity values. If no value is passed
        then positive infinity values will be replaced with a very large number.
    neginf
        Value to be used to fill negative infinity values.
        If no value is passed then negative infinity values
        will be replaced with a very small (or negative) number.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with the non-finite values replaced.
        If copy is False, this may be x itself.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, nan])
    >>> ivy.nan_to_num(x)
    ivy.array([1.,    1.,   3.,   0.0])
    >>> x = ivy.array([1, 2, 3, inf])
    >>> ivy.nan_to_num(x, posinf=5e+100)
    ivy.array([1.,   2.,   3.,   5e+100])
    """
    return ivy.current_backend(x).nan_to_num(
        x,
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
        out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def logaddexp2(
    x1: Union[ivy.Array, ivy.NativeArray, float, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, float, list, tuple],    
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculates log2(2**x1 + 2**x2).

    Parameters
    ----------
    x1
        First array-like input.
    x2
        Second array-input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Element-wise logaddexp2 of x1 and x2.

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> ivy.logaddexp2(x1, x2)
    ivy.array([4.169925, 5.169925, 6.169925])
    """
    return ivy.current_backend(x1, x2).logaddexp2(x1, x2, out=out)
