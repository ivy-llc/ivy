# local
from typing import Optional, Union, Tuple, List
from numbers import Number
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
        first input array, must be integers
    x2
        second input array, must be integers
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
        Second input array.
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
    ivy.array([ 0.,  0.,  nan])
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
def copysign(
    x1: Union[ivy.Array, ivy.NativeArray, Number],
    x2: Union[ivy.Array, ivy.NativeArray, Number],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Change the signs of x1 to match x2
    x1 and x2 must be broadcastable to a common shape

    Parameters
    ----------
    x1
        Array or scalar to change the sign of
    x2
        Array or scalar from which the new signs are applied
        Unsigned zeroes are considered positive.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        x1 with the signs of x2.
        This is a scalar if both x1 and x2 are scalars.

    Examples
    --------
    >>> x1 = ivy.array([-1, 0, 23, 2])
    >>> x2 = ivy.array([1, -1, -10, 44])
    >>> ivy.copysign(x1, x2)
    ivy.array([  1.,  -0., -23.,   2.])
    >>> ivy.copysign(x1, -1)
    ivy.array([ -1.,  -0., -23.,  -2.])
    >>> ivy.copysign(-10, 1)
    ivy.array(10.)
    """
    return ivy.current_backend().copysign(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def count_nonzero(
    a: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[Union[int, ivy.Array]] = None,
) -> ivy.Array:
    """Counts the number of non-zero values in the array a.

    Parameters
    ----------
    a
        array for which to count non-zeros.
    axis
        optional axis or tuple of axes along which to count non-zeros. Default is
        None, meaning that non-zeros will be counted along a flattened
        version of the input array.
    keepdims
        optional, if this is set to True, the axes that are counted are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.
    dtype
        optional output dtype. Default is of type integer.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Number of non-zero values in the array along a given axis. Otherwise,
        the total number of non-zero values in the array is returned.

    Examples
    --------
    >>> a = ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]])
    >>> ivy.count_nonzero(a)
    ivy.array(7)
    >>> a = ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]])
    >>> ivy.count_nonzero(a, axis=0)
    ivy.array([1, 2, 2, 2])
    >>> a = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> ivy.count_nonzero(a, axis=(0,1), keepdims=True)
    ivy.array([[[3, 4]]])
    """
    return ivy.current_backend().count_nonzero(
        a, axis=axis, keepdims=keepdims, dtype=dtype, out=out
    )


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
        x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def signbit(
    x: Union[ivy.Array, ivy.NativeArray, float, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Output array, or reference to out if that was supplied.
        This is a scalar if x is a scalar.

    Examples
    --------
    >>> x = ivy.array([1, -2, 3])
    >>> ivy.signbit(x)
    ivy.array([False, True, False])
    """
    return ivy.current_backend(x).signbit(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def diff(
    x: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the n-th discrete difference along the given axis.

    Parameters
    ----------
    x
        array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Rreturns the n-th discrete difference along the given axis.

    Examples
    --------
    >>> x = ivy.array([1, 2, 4, 7, 0])
    >>> ivy.diff(x)
    ivy.array([ 1,  2,  3, -7])
    """
    return ivy.current_backend().diff(x, out=out)


@handle_exceptions
def allclose(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Returns a True if the two arrays are element-wise equal
    within given tolerance; otherwise False.
    The tolerance values are positive, typically very small numbers.
    The relative difference (rtol * abs(x2)) and the absolute difference
    atol are added together to compare against the absolute difference
    between x1 and x2.
    The default atol is not appropriate for comparing numbers that are
    much smaller than one

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array.
    rtol
        The relative tolerance parameter.
    atol
        The absolute tolerance parameter.
    equal_nan
        Whether to compare NaN's as equal. If True, NaN's in x1 will be
        considered equal to NaN's in x2 in the output array.
    out
        Alternate output array in which to place the result.
        The default is None.

    Returns
    -------
    ret
        Returns True if the two arrays are equal within the given tolerance;
        False otherwise.

    Examples
    --------
    >>> x1 = ivy.array([1e10, 1e-7])
    >>> x2 = ivy.array([1.00001e10, 1e-8])
    >>> y = ivy.allclose(x1, x2)
    >>> print(y)
    False

    >>> x1 = ivy.array([1.0, ivy.nan])
    >>> x2 = ivy.array([1.0, ivy.nan])
    >>> y = ivy.allclose(x1, x2, equal_nan=True)
    >>> print(y)
    True

    >>> x1 = ivy.array([1e-10, 1e-10])
    >>> x2 = ivy.array([1.00001e-10, 1e-10])
    >>> y = ivy.allclose(x1, x2, rtol=0.005, atol=0.0)
    >>> print(y)
    True

    """
    return ivy.current_backend().allclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def fix(
    x: Union[ivy.Array, ivy.NativeArray, float, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters
    ----------
    x
        Array input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array of floats with elements corresponding to input elements
        rounded to nearest integer towards zero, element-wise.

    Examples
    --------
    >>> x = ivy.array([2.1, 2.9, -2.1])
    >>> ivy.fix(x)
    ivy.array([ 2.,  2., -2.])
    """
    return ivy.current_backend(x).fix(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nextafter(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Return the next floating-point value after x1 towards x2, element-wise.

    Parameters
    ----------
    x1
        First input array.
    x2
        Second input array.
    out
        Alternate output array in which to place the result.
        The default is None.

    Returns
    -------
    ret
        The next representable values of x1 in the direction of x2.

    Examples
    --------
    >>> x1 = ivy.array([1.0e-50, 2.0e+50])
    >>> x2 = ivy.array([2.0, 1.0])
    >>> ivy.nextafter(x1, x2)
    ivy.array([1.4013e-45., 3.4028e+38])
    """
    return ivy.current_backend(x1, x2).nextafter(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def zeta(
    x: Union[ivy.Array, ivy.NativeArray],
    q: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Compute the Hurwitz zeta function.

    Parameters
    ----------
    x
        First input array.
    q
        Second input array.
    out
        Alternate output array in which to place the result.
        The default is None.

    Returns
    -------
    ret
        Array with values computed from zeta function from
        input arrays' values.

    Examples
    --------
    >>> x = ivy.array([5.0, 3.0])
    >>> q = ivy.array([2.0])
    >>> ivy.zeta(x, q)
    ivy.array([0.0369, 0.2021])
    """
    return ivy.current_backend(x, q).zeta(x, q, out=out)


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
def gradient(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    spacing: Optional[Union[int, list, tuple]] = 1,
    edge_order: Optional[int] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
) -> Union[ivy.Array, List[ivy.Array]]:
    """Calculates gradient of x with respect to (w.r.t.) spacing

    Parameters
    ----------
    x
        input array representing outcomes of the function
    spacing
        if not given, indices of x will be used
        if scalar indices of x will be scaled with this value
        if array gradient of x w.r.t. spacing
    edge_order
        1 or 2, for 'frist order' and 'second order' estimation
        of boundary values of gradient respectively.
        Note: jax supports edge_order=1 case only
    axis
        dimension(s) to approximate the gradient over
        by default partial gradient is computed in every dimention

    Returns
    -------
    ret
        Array with values computed from gradient function from
        inputs

    Examples
    --------
    >>> spacing = (ivy.array([-2., -1., 1., 4.]),)
    >>> x = ivy.array([4., 1., 1., 16.], )
    >>> ivy.gradient(x, spacing=spacing)
    ivy.array([-3., -2.,  2.,  5.])

    >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> ivy.gradient(x)
    [ivy.array([[ 9., 18., 36., 72.],
       [ 9., 18., 36., 72.]]), ivy.array([[ 1. ,  1.5,  3. ,  4. ],
       [10. , 15. , 30. , 40. ]])]

    >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> ivy.gradient(x, spacing=2.0)
    [ivy.array([[ 4.5,  9. , 18. , 36. ],
       [ 4.5,  9. , 18. , 36. ]]), ivy.array([[ 0.5 ,  0.75,  1.5 ,  2.  ],
       [ 5.  ,  7.5 , 15.  , 20.  ]])]

    >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> ivy.gradient(x, axis=1)
    ivy.array([[ 1. ,  1.5,  3. ,  4. ],
       [10. , 15. , 30. , 40. ]])

    >>> x = ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]])
    >>> ivy.gradient(x, spacing=[3., 2.])
    [ivy.array([[ 3.,  6., 12., 24.],
       [ 3.,  6., 12., 24.]]), ivy.array([[ 0.5 ,  0.75,  1.5 ,  2.  ],
       [ 5.  ,  7.5 , 15.  , 20.  ]])]

    >>> spacing = (ivy.array([0, 2]), ivy.array([0, 3, 6, 9]))
    >>> ivy.gradient(x, spacing=spacing)
    [ivy.array([[ 4.5,  9. , 18. , 36. ],
       [ 4.5,  9. , 18. , 36. ]]), ivy.array([[ 0.33333333, 0.5,  1., 1.33333333],
       [ 3.33333333,  5.        , 10.        , 13.33333333]])]

    """
    return ivy.current_backend(x).gradient(
        x, spacing=spacing, edge_order=edge_order, axis=axis
    )
