# global
from typing import Union, Tuple, Optional

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

# noinspection PyShadowingBuiltins
def min(x: Union[ivy.Array, ivy.NativeArray],
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
        -> ivy.Array:
    """
    Calculates the minimum value of the input array x.
    
    .. note::
    When the number of elements over which to compute the minimum value is zero, 
    the minimum value is implementation-defined. Specification-compliant libraries may choose to raise 
    an error, return a sentinel value (e.g., if x is a floating-point input array, return NaN), or return 
    the maximum possible value for the input array x data type (e.g., if x is a floating-point array, 
    return +infinity).

    **Special Cases**
    For floating-point operands,

    If x_i is NaN, the minimum value is NaN (i.e., NaN values propagate).    
  
    Parameters
    ----------
    x:
        Input array containing elements to min.
    axis:
         axis or axes along which minimum values must be computed.
         By default, the minimum value must be computed over the entire array.
         If a tuple of integers, minimum values must be computed over multiple axes. Default: None.
    keepdims:
        optional boolean, if True, the reduced axes (dimensions) must be included in the result as 
        singleton dimensions, and, accordingly, the result must be compatible with
        the input array (see Broadcasting). 
        Otherwise, if False, the reduced axes (dimensions) must not be included in the result. 
        Default: False.
    out: 
        optional output array, for writing the result to.
    
    Returns
    ----------
    return: 
        if the minimum value was computed over the entire array, a zero-dimensional array containing the 
        minimum value; otherwise, a non-zero-dimensional array containing the minimum values. 
        The returned array must have the same data type as x.
    """
    return _cur_framework.min(x, axis, keepdims, out)


# noinspection PyShadowingBuiltins
def max(x: Union[ivy.Array, ivy.NativeArray],
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None) \
        -> ivy.Array:
    """
    Calculates the maximum value of the input array ``x``.
    .. note::
       When the number of elements over which to compute the maximum value is zero, 
       the maximum value is implementation-defined. Specification-compliant libraries may choose to raise 
       an error, return a sentinel value (e.g., if ``x`` is a floating-point input array, return ``NaN``), 
       or return the minimum possible value for the input array ``x`` data type (e.g., if ``x`` is a 
       floating-point array, return ``-infinity``).

    **Special Cases**
    For floating-point operands,
    -   If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values propagate).
    
    Parameters
    ----------
    x: 
        input array. Should have a numeric data type.
    axis: 
        axis or axes along which maximum values must be computed. By default, the maximum value must be computed over the entire array. If a tuple of integers, maximum values must be computed over multiple axes. Default: ``None``.
    keepdims:
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, 
        and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). 
        Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. 
        Default: ``False``.
    out: 
        optional output array, for writing the result to.
    Returns
    -------
    return: 
        if the maximum value was computed over the entire array, a zero-dimensional array containing the 
        maximum value; otherwise, a non-zero-dimensional array containing the maximum values. 
        The returned array must have the same data type as ``x``.
    """
    return _cur_framework.max(x, axis, keepdims,out=out)



def var(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) \
        -> ivy.Array:
    """
    Calculates the variance of the input array x.
    :param x: input array
    :param axis: axis or axes along which variances must be computed. By default, the variance must be computed over the entire array
    :param correction: degrees of freedom adjustment
    :param keepdims: Default: False
    :return: The returned array must have the same data type as x.
    """
    return _cur_framework(x).var(x, axis, correction, keepdims)


def mean(x: Union[ivy.Array, ivy.NativeArray],
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False)\
        -> ivy.Array:
    """
    Calculates the arithmetic mean of the input array ``x``.
    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the arithmetic mean.
    -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
    -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN`` values propagate).
    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which arithmetic means must be computed. By default, the mean must be computed over the entire array. If a tuple of integers, arithmetic means must be computed over multiple axes. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.
    Returns
    -------
    out: array
        if the arithmetic mean was computed over the entire array, a zero-dimensional array containing the arithmetic mean; otherwise, a non-zero-dimensional array containing the arithmetic means. The returned array must have the same data type as ``x``.
        .. note::
           While this specification recommends that this function only accept input arrays having a floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default floating-point data type.
    """
    return _cur_framework(x).mean(x, axis, keepdims)


def prod(x: Union[ivy.Array, ivy.NativeArray],
         axis: Optional[Union[int, Tuple[int]]] = None,
         dtype: Optional[Union[ivy.Dtype, str]] = None,
         keepdims: bool = False)\
        -> ivy.Array:
    """
    Calculates the product of input array x elements.

    :param x: input array. Should have a numeric data type.
    :type x: array
    :param axis: axis or axes along which products must be computed. By default, the product must be
     computed over the entire array. If a tuple of integers, products must be computed over multiple axes. Default: None.
    :type axis: Union[int, Tuple[int, ...]
    :param dtype: data type of the returned array. If None,
        if the default data type corresponding to the data type “kind” (integer or floating-point) of x has a smaller 
         range of values than the data type of x (e.g., x has data type int64 and the default data type is int32,
         or x has data type uint64 and the default data type is int64), the returned array must have the same data type as x.
        if x has a floating-point data type, the returned array must have the default floating-point data type.
        if x has a signed integer data type (e.g., int16), the returned array must have the default integer data type.
        if x has an unsigned integer data type (e.g., uint16), the returned array must have an unsigned integer data type
         having the same number of bits as the default integer data type (e.g., if the default integer data type is int32, 
         the returned array must have a uint32 data type).
        If the data type (either specified or resolved) differs from the data type of x, the input array should be cast 
        to the specified data type before computing the product. Default: None.
    :type dtype: dtype
    :param keepdims: if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, 
    accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False, the reduced axes 
    (dimensions) must not be included in the result. Default: False.
    :type keepdims: bool

    Returns
    :return out: if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, 
    a non-zero-dimensional array containing the products. The returned array must have a data type as described by the dtype 
    parameter above.
    :type out: array
    """
    return _cur_framework.prod(x, axis, dtype, keepdims)


def sum(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Optional[Union[ivy.Dtype, str]] = None,
        keepdims: bool = False) -> ivy.Array:
    """
    Calculates the sum of the input array ``x``.
    **Special Cases**
    Let ``N`` equal the number of elements over which to compute the sum.
    -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).
    For floating-point operands,
    -   If ``x_i`` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).
    Parameters
    ----------
    x:
        input array. Should have a numeric data type.
    axis:
        axis or axes along which sums must be computed. By default, the sum must be computed over the entire array. If a tuple of integers, sums must be computed over multiple axes. Default: ``None``.
    dtype:
        data type of the returned array. If ``None``,
        -   if the default data type corresponding to the data type "kind" (integer or floating-point) of ``x`` has a smaller range of values than the data type of ``x`` (e.g., ``x`` has data type ``int64`` and the default data type is ``int32``, or ``x`` has data type ``uint64`` and the default data type is ``int64``), the returned array must have the same data type as ``x``.
        -   if ``x`` has a floating-point data type, the returned array must have the default floating-point data type.
        -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
        -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).
        If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum. Default: ``None``.
        .. note::
           keyword argument is intended to help prevent data type overflows.
    keepdims:
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.
    Returns
    -------
    out:
        if the sum was computed over the entire array, a zero-dimensional array containing the sum; otherwise, an array containing the sums. The returned array must have a data type as described by the ``dtype`` parameter above.
    """

    return _cur_framework(x).sum(x, axis, dtype, keepdims)

def std(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False)\
        -> ivy.Array:
    """
    Computes the arithmetic standard deviation along a given axis. The standard deviation is taken over
    the flattened array by default, otherwise over the specified axis.

    :param x: Array containing numbers whose standard deviation is desired.
    :type x: array
    :param axis: Axis or axes along which the means are computed. The default is to compute the mean of the flattened
                    array. If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis
                    or all the axes as before.
    :type axis: int or sequence of ints
    :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size
                        one. With this option, the result will broadcast correctly against the input array.
    :type keepdims: bool, optional
    :return: The array with standard deviations computed.
    """
    return _cur_framework(x).std(x, axis, correction, keepdims)


# Extra #
# ------#

def einsum(equation, *operands):
    """
    Sums the product of the elements of the input operands along dimensions specified using a notation based on the
    Einstein summation convention.

    :param equation: A str describing the contraction, in the same format as numpy.einsum.
    :type equation: str
    :param operands: the inputs to contract (each one an ivy.Array), whose shapes should be consistent with equation.
    :type operands: seq of arrays
    :return: The array with sums computed.
    """
    return _cur_framework(operands[0]).einsum(equation, *operands)
