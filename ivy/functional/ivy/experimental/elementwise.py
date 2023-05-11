# local
from typing import Optional, Union, Tuple, List
from numbers import Number
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@integer_arrays_to_float
def sinc(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate an implementation-dependent approximation of the principal value of
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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def lcm(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the element-wise least common multiple (LCM) of x1 and x2.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def fmax(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the element-wise maximums of two arrays. Differs from ivy.maximum in the
    case where one of the elements is NaN. ivy.maximum returns the NaN element while
    ivy.fmax returns the non-NaN element.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def fmin(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the element-wise minimums of two arrays. Differs from ivy.minimum in the
    case where one of the elements is NaN. ivy.minimum returns the NaN element while
    ivy.fmin returns the non-NaN element.

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
        Array with element-wise minimums.

    Examples
    --------
    >>> x1 = ivy.array([2, 3, 4])
    >>> x2 = ivy.array([1, 5, 2])
    >>> ivy.fmin(x1, x2)
    ivy.array([1, 3, 2])

    >>> x1 = ivy.array([ivy.nan, 0, ivy.nan])
    >>> x2 = ivy.array([0, ivy.nan, ivy.nan])
    >>> ivy.fmin(x1, x2)
    ivy.array([ 0.,  0., nan])
    """
    return ivy.current_backend().fmin(x1, x2, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def trapz(
    y: ivy.Array,
    /,
    *,
    x: Optional[ivy.Array] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Integrate along the given axis using the composite trapezoidal rule.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def float_power(
    x1: Union[ivy.Array, float, list, tuple],
    x2: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Raise each base in x1 to the positionally-corresponding power in x2. x1 and x2 must
    be broadcastable to the same shape. This differs from the power function in that
    integers, float16, and float32 are promoted to floats with a minimum precision of
    float64 so that the result is always inexact.

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


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def exp2(
    x: Union[ivy.Array, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate 2**p for all p in the input array.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_exceptions
def copysign(
    x1: Union[ivy.Array, ivy.NativeArray, Number],
    x2: Union[ivy.Array, ivy.NativeArray, Number],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Change the signs of x1 to match x2 x1 and x2 must be broadcastable to a common
    shape.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def count_nonzero(
    a: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Count the number of non-zero values in the array a.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def nansum(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as
    zero.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def gcd(
    x1: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the greatest common divisor of |x1| and |x2|.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def isclose(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a boolean array where two arrays are element-wise equal within a tolerance.

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


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def angle(
    z: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    deg: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate Element-wise the angle for an array of complex numbers(x+yj).

    Parameters
    ----------
    z
        Array-like input.
    deg
        optional bool.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns an array of angles for each complex number in the input.
        If deg is False(default), angle is calculated in radian and if
        deg is True, then angle is calculated in degrees.

    Examples
    --------
    >>> ivy.set_backend('tensorflow')
    >>> z = ivy.array([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> z
    ivy.array([-1.+1.j, -2.+2.j,  3.-3.j])
    >>> ivy.angle(z)
    ivy.array([ 2.35619449,  2.35619449, -0.78539816])
    >>> ivy.set_backend('numpy')
    >>> ivy.angle(z,deg=True)
    ivy.array([135., 135., -45.])
    """
    return ivy.current_backend(z).angle(z, deg=deg, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def imag(
    val: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the Imaginary part of a complex numbers(x+yj).

    Parameters
    ----------
    val
        Array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns an array with the imaginary part of complex numbers.

    Examples
    --------
    >>> b = ivy.array(np.array([1+2j, 3+4j, 5+6j]))
    >>> b
    ivy.array([1.+2.j, 3.+4.j, 5.+6.j])
    >>> ivy.imag(b)
    ivy.array([2., 4., 6.])
    """
    return ivy.current_backend(val).imag(val, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def nan_to_num(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Replace NaN with zero and infinity with large finite numbers (default behaviour) or
    with the numbers defined by the user using the nan, posinf and/or neginf keywords.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def logaddexp2(
    x1: Union[ivy.Array, ivy.NativeArray, float, list, tuple],
    x2: Union[ivy.Array, ivy.NativeArray, float, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate log2(2**x1 + 2**x2).

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


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def signbit(
    x: Union[ivy.Array, ivy.NativeArray, float, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return element-wise True where signbit is set (less than zero).

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def hypot(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Return the hypotenuse given the two sides of a right angle triangle.

    Parameters
    ----------
    x1
        The first input array
    x2
        The second input array

    Returns
    -------
    ret
        An array with the hypotenuse

    Examples
    --------
    >>> a = ivy.array([3.0, 4.0, 5.0])
    >>> b = ivy.array([4.0, 5.0, 6.0])
    >>> ivy.hypot(a, b)
    ivy.array([5.0, 6.4031, 7.8102])
    """
    return ivy.current_backend(x1, x2).hypot(x1, x2, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def diff(
    x: Union[ivy.Array, ivy.NativeArray, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[ivy.Array, ivy.NativeArray, int, list, tuple]] = None,
    append: Optional[Union[ivy.Array, ivy.NativeArray, int, list, tuple]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the n-th discrete difference along the given axis.

    Parameters
    ----------
    x
        Array-like input.
    n
        The number of times values are differenced. If zero, the input is returned
        as-is.
    axis
        The axis along which the difference is taken, default is the last axis.
    prepend,append
        Values to prepend/append to x along given axis prior to performing the
        difference. Scalar values are expanded to arrays with length 1 in the direction
        of axis and the shape of the input array in along all other axes. Otherwise the
        dimension and shape must match x except along axis.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Returns the n-th discrete difference along the given axis.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.array([1, 2, 4, 7, 0])
    >>> ivy.diff(x)
    ivy.array([ 1,  2,  3, -7])
    """
    return ivy.current_backend().diff(
        x, n=n, axis=axis, prepend=prepend, append=append, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def allclose(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Return a True if the two arrays are element-wise equal within given tolerance;
    otherwise False.

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
    ivy.array(False)

    >>> x1 = ivy.array([1.0, ivy.nan])
    >>> x2 = ivy.array([1.0, ivy.nan])
    >>> y = ivy.allclose(x1, x2, equal_nan=True)
    >>> print(y)
    ivy.array(True)

    >>> x1 = ivy.array([1e-10, 1e-10])
    >>> x2 = ivy.array([1.00001e-10, 1e-10])
    >>> y = ivy.allclose(x1, x2, rtol=0.005, atol=0.0)
    >>> print(y)
    ivy.array(True)
    """
    return ivy.current_backend().allclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def fix(
    x: Union[ivy.Array, ivy.NativeArray, float, int, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Round an array of floats element-wise to nearest integer towards zero. The rounded
    values are returned as floats.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def zeta(
    x: Union[ivy.Array, ivy.NativeArray],
    q: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Compute the Hurwitz zeta function elementwisely with each pair of floats in two
    arrays.

    Parameters
    ----------
    x
        First input array.
    q
        Second input array, must have the same shape as the first input array
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
    >>> q = ivy.array([2.0, 2.0])
    >>> ivy.zeta(x, q)
    ivy.array([0.0369, 0.2021])
    """
    return ivy.current_backend(x, q).zeta(x, q, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def gradient(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    spacing: Union[int, list, tuple] = 1,
    edge_order: int = 1,
    axis: Optional[Union[int, list, tuple]] = None,
) -> Union[ivy.Array, List[ivy.Array]]:
    """
    Calculate gradient of x with respect to (w.r.t.) spacing.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_exceptions
def xlogy(
    x: Union[ivy.Array, ivy.NativeArray],
    y: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> bool:
    """
    Compute x*log(y) element-wise so that the result is 0 if x = 0.

    Parameters
    ----------
    x
        First input array.
    y
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
    >>> x = ivy.zeros(3)
    >>> y = ivy.array([-1.0, 0.0, 1.0])
    >>> ivy.xlogy(x, y)
    ivy.array([0.0, 0.0, 0.0])

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([3.0, 2.0, 1.0])
    >>> ivy.xlogy(x, y)
    ivy.array([1.0986, 1.3863, 0.0000])
    """
    return ivy.current_backend(x, y).xlogy(x, y, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def real(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test each element ``x_i`` of the input array ``x`` to
    take only real part from it.
    Returns a float array, where it only contains .
    If element has complex type with zero complex part, the return value
    will be that element, else it only returns real part.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing test results. An element ``out_i`` is
        ``real number`` if ``x_i`` contain real number part only
        and if it is ``real number with complex part also`` then it
        returns the real number part.
        The returned array should have a data type of ``float``.

    The descriptions above assume an array input for simplicity, but
    the method also accepts :class:`ivy.Container` instances
    in place of: class:`ivy.Array` or :class:`ivy.NativeArray`
    instances, as shown in the type hints and also the examples below.

    Examples
    --------
    With :class:`ivy.Array` inputs:
    >>> x = ivy.array([[[1.1], [2], [-6.3]]])
    >>> z = ivy.real(x)
    >>> print(z)
    ivy.array([[[1.1], [2.], [-6.3]]])

    >>> x = ivy.array([4.2-0j, 3j, 7+5j])
    >>> z = ivy.real(x)
    >>> print(z)
    ivy.array([4.2, 0., 7.])

    With :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([-6.7-7j, 0.314+0.355j, 1.23]),\
                          b=ivy.array([5j, 5.32-6.55j, 3.001]))
    >>> z = ivy.real(x)
    >>> print(z)
    {
        a: ivy.array([-6.7, 0.314, 1.23]),
        b: ivy.array([0., 5.32, 3.001])
    }
    """
    return ivy.current_backend(x).real(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_ivy_arrays
def binarizer(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: float = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Map the values of the input tensor to either 0 or 1, element-wise, based on the
    outcome of a comparison against a threshold value.

    Parameters
    ----------
    x
        Data to be binarized
    threshold
        Values greater than this are
        mapped to 1, others to 0.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        Binarized output data
    """
    xc = ivy.copy_array(x, out=out)
    if ivy.is_bool_dtype(xc) and ivy.current_backend_str() == "torch":
        xc = ivy.astype(xc, ivy.default_float_dtype())
    if ivy.is_complex_dtype(xc):
        xc = ivy.abs(xc)
    ret = ivy.where(xc > threshold, 1, 0)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def conj(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the complex conjugate of complex values in x.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        an arrray of the same dtype as the input array with
        the complex conjugates of the complex values present
        in the input array. If x is a scalar then a scalar
        will be returned.

    The descriptions above assume an array input for simplicity, but
    the method also accepts :class:`ivy.Container` instances
    in place of: class:`ivy.Array` or :class:`ivy.NativeArray`
    instances, as shown in the type hints and also the examples below.


    Examples
    --------
    With :class:`ivy.Array` inputs:
    >>> x = ivy.array([4.2-0j, 3j, 7+5j])
    >>> z = ivy.conj(x)
    >>> print(z)
    ivy.array([4.2+0j, -3j, 7+5j])

    With :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([-6.7-7j, 0.314+0.355j, 1.23]),\
                          b=ivy.array([5j, 5.32-6.55j, 3.001]))
    >>> z = ivy.conj(x)
    >>> print(z)
    {
        a: ivy.array([-6.7+7j, 0.314-0.355j, 1.23]),
        b: ivy.array([-5j, 5.32+6.55j, 3.001])
    }
    """
    return ivy.current_backend(x).conj(x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def ldexp(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return x1 * (2**x2), element-wise.

    Parameters
    ----------
    x1
        Input array.
    x2
        Input array.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The next representable values of x1 in the direction of x2.

    Examples
    --------
    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([0, 1, 2])
    >>> ivy.ldexp(x1, x2)
    ivy.array([1, 4, 12])
    """
    return ivy.current_backend(x1, x2).ldexp(x1, x2, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def lerp(
    input: Union[ivy.Array, ivy.NativeArray],
    end: Union[ivy.Array, ivy.NativeArray],
    weight: Union[ivy.Array, ivy.NativeArray, float],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a linear interpolation of two arrays start (given by input) and end.

    based on a scalar or array weight.
        input + weight * (end - input),  element-wise.

    Parameters
    ----------
    input
        array of starting points
    end
        array of ending points
    weight
        the weight for the interpolation formula. Scalar or Array.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The result of  input + ((end - input) * weight)

    Examples
    --------
    With :class:`ivy.Array` inputs:
    >>> input = ivy.array([1, 2, 3])
    >>> end = ivy.array([10, 10, 10])
    >>> weight = 0.5
    >>> ivy.lerp(input, end, weight)
    ivy.array([5.5, 6. , 6.5])
    >>> input = ivy.array([1.1, 1.2, 1.3])
    >>> end = ivy.array([20])
    >>> weight = ivy.array([0.4, 0.5, 0.6])
    >>> y = ivy.zeros(3)
    >>> ivy.lerp(input, end, weight, out=y)
    ivy.array([ 8.65999985, 10.59999943, 12.52000141])
    >>> input = ivy.array([[4, 5, 6],[4.1, 4.2, 4.3]])
    >>> end = ivy.array([10])
    >>> weight = ivy.array([0.5])
    >>> ivy.lerp(input, end, weight, out=input)
    ivy.array([[7.        , 7.5       , 8.        ],
    ...       [7.05000019, 7.0999999 , 7.1500001 ]])
    With :class:`ivy.Container` input:
    >>> input = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> end = ivy.array([10.])
    >>> weight = 1.1
    >>> y = input.lerp(end, weight)
    >>> print(y)
    {
        a: ivy.array([11., 10.90000057, 10.80000019]),
        b: ivy.array([10.70000076, 10.60000038, 10.5])
    }
    >>> input = ivy.Container(a=ivy.array([10.1, 11.1]), b=ivy.array([10, 11]))
    >>> end = ivy.Container(a=ivy.array([5]), b=ivy.array([0]))
    >>> weight = ivy.Container(a=0.5)
    >>> y = input.lerp(end, weight)
    >>> print(y)
    {
        a: ivy.array([7.55000019, 8.05000019]),
        b: {
            a: ivy.array([5., 5.5])
        }
    }
    """
    input_end_allowed_types = [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex",
    ]
    weight_allowed_types = ["float16", "bfloat16", "float32", "float64"]

    if not ivy.is_array(input):
        input = ivy.array([input])
    if not ivy.is_array(end):
        end = ivy.array([end])
    if (
        ivy.dtype(input) not in input_end_allowed_types
        or ivy.dtype(end) not in input_end_allowed_types
    ):
        input = ivy.astype(input, "float64")
        end = ivy.astype(end, "float64")

    if ivy.is_array(weight):
        if ivy.dtype(weight) not in weight_allowed_types:
            weight = ivy.astype(weight, "float64")
    else:
        if not isinstance(weight, float):
            weight = ivy.astype(ivy.array([weight]), "float64")

    return ivy.add(input, ivy.multiply(weight, ivy.subtract(end, input)), out=out)


lerp.mixed_function = True


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def frexp(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Tuple[ivy.Array, ivy.Array]] = None,
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Decompose the elements of x into mantissa and twos exponent.

    Parameters
    ----------
    x
        Input array.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        A tuple of two arrays, the mantissa and the twos exponent.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> ivy.frexp(x)
    (ivy.array([0.5, 0.5, 0.75]), ivy.array([1, 2, 2]))
    """
    return ivy.current_backend(x).frexp(x, out=out)
