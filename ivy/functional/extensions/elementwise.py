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
