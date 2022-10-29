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
