# global
from typing import Union, Tuple, Optional

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

# noinspection PyShadowingBuiltins
def min(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the minimum value of the input array x.

    .. note::
    When the number of elements over which to compute the minimum value is zero, the
    minimum value is implementation-defined. Specification-compliant libraries may
    choose to raise an error, return a sentinel value (e.g., if x is a floating-point
    input array, return NaN), or return the maximum possible value for the input array x
    data type (e.g., if x is a floating-point array, return +infinity).

    **Special Cases**

    For floating-point operands,

    If x_i is NaN, the minimum value is NaN (i.e., NaN values propagate).

    Parameters
    ----------
    x
        Input array containing elements to min.
    axis
         axis or axes along which minimum values must be computed. By default, the
         minimum value must be computed over the entire array. If a tuple of integers,
         minimum values must be computed over multiple axes. Default: None.
    keepdims
        optional boolean, if True, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be compatible
        with the input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: False.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the minimum value was computed over the entire array, a zero-dimensional
        array containing the minimum value; otherwise, a non-zero-dimensional array
        containing the minimum values. The returned array must have the same data type
        as x.

    """
    return _cur_framework.min(x, axis, keepdims, out)


# noinspection PyShadowingBuiltins
def max(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the maximum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the maximum value is zero, the
       maximum value is implementation-defined. Specification-compliant libraries may
       choose to raise an error, return a sentinel value (e.g., if ``x`` is a
       floating-point input array, return ``NaN``), or return the minimum possible value
       for the input array ``x`` data type (e.g., if ``x`` is a floating-point array,
       return ``-infinity``).

    **Special Cases**

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which maximum values must be computed. By default, the
        maximum value must be computed over the entire array. If a tuple of integers,
        maximum values must be computed over multiple axes. Default: ``None``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the maximum value was computed over the entire array, a zero-dimensional
        array containing the maximum value; otherwise, a non-zero-dimensional array
        containing the maximum values. The returned array must have the same data type
        as ``x``.

    """
    return _cur_framework.max(x, axis, keepdims, out=out)


def var(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the variance of the input array x.

    **Special Cases**

    Let N equal the number of elements over which to compute the variance.

    If N - correction is less than or equal to 0, the variance is NaN.

    If x_i is NaN, the variance is NaN (i.e., NaN values propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which variances must be computed. By default, the variance
        must be computed over the entire array. If a tuple of integers, variances must
        be computed over multiple axes. Default: None.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other than 0
        has the effect of adjusting the divisor during the calculation of the variance
        according to N-c where N corresponds to the total number of elements over which
        the variance is computed and c corresponds to the provided degrees of freedom
        adjustment. When computing the variance of a population, setting this parameter
        to 0 is the standard choice (i.e., the provided array contains data constituting
        an entire population). When computing the unbiased sample variance, setting this
        parameter to 1 is the standard choice (i.e., the provided array contains data
        sampled from a larger population; this is commonly referred to as Bessel's
        correction). Default: 0.
    keepdims
        if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: False.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the variance was computed over the entire array, a zero-dimensional array
        containing the variance; otherwise, a non-zero-dimensional array containing the
        variances. The returned array must have the same data type as x.

    """
    return _cur_framework(x).var(x, axis, correction, keepdims, out=out)


def mean(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the arithmetic mean of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the arithmetic mean.
    -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
    -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which arithmetic means must be computed. By default, the mean
        must be computed over the entire array. If a tuple of integers, arithmetic means
        must be computed over multiple axes. Default: ``None``.
    keepdims
        bool, if ``True``, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible with
        the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array, if the arithmetic mean was computed over the entire array, a
        zero-dimensional array containing the arithmetic mean; otherwise, a
        non-zero-dimensional array containing the arithmetic means. The returned array
        must have the same data type as ``x``.
        .. note::
           While this specification recommends that this function only accept input
           arrays having a floating-point data type, specification-compliant array
           libraries may choose to accept input arrays having an integer data type.
           While mixed data type promotion is implementation-defined, if the input array
           ``x`` has an integer data type, the returned array must have the default
           floating-point data type.

    """
    return _cur_framework(x).mean(x, axis, keepdims, out=out)


def prod(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the product of input array x elements.

    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which products must be computed. By default, the product must
        be computed over the entire array. If a tuple of integers, products must be
        computed over multiple axes. Default: None.
    keepdims
        bool, if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: False.
    dtype
        data type of the returned array. If None,
        if the default data type corresponding to the data type “kind” (integer or
        floating-point) of x has a smaller range of values than the data type of x
        (e.g., x has data type int64 and the default data type is int32, or x has data
        type uint64 and the default data type is int64), the returned array must have
        the same data type as x. if x has a floating-point data type, the returned array
        must have the default floating-point data type. if x has a signed integer data
        type (e.g., int16), the returned array must have the default integer data type.
        if x has an unsigned integer data type (e.g., uint16), the returned array must
        have an unsigned integer data type having the same number of bits as the default
        integer data type (e.g., if the default integer data type is int32, the returned
        array must have a uint32 data type). If the data type (either specified or
        resolved) differs from the data type of x, the input array should be cast to the
        specified data type before computing the product. Default: None.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array,  if the product was computed over the entire array, a zero-dimensional
        array containing the product; otherwise, a non-zero-dimensional array containing
        the products. The returned array must have a data type as described by the dtype
        parameter above.

    """
    return _cur_framework.prod(x, axis, keepdims, dtype=dtype, out=out)


def sum(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the sum of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the sum.
    -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

    For floating-point operands,
    -   If ``x_i`` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.
    axis
        Axis or axes along which sums must be computed. By default, the sum must be
        computed over the entire array. If a tuple of integers, sums must be computed
        over multiple axes. Default: ``None``.
    dtype
        Data type of the returned array. If ``None``,
        -   If the default data type corresponding to the data type "kind" (integer or
            floating-point) of ``x`` has a smaller range of values than the data type of
            ``x`` (e.g., ``x`` has data type ``int64`` and the default data type is
            ``int32``, or ``x`` has data type ``uint64`` and the default data type is
            ``int64``), the returned array must have the same data type as ``x``.
        -   If ``x`` has a floating-point data type, the returned array must have the
            default floating-point data type.
        -   If ``x`` has a signed integer data type (e.g., ``int16``), the returned
            array must have the default integer data type.
        -   If ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned
            array must have an unsigned integer data type having the same number of bits
            as the default integer data type (e.g., if the default integer data type is
            ``int32``, the returned array must have a ``uint32`` data type).

        If the data type (either specified or resolved) differs from the data type of
        ``x``, the input array should be cast to the specified data type before
        computing the sum. Default: ``None``.

        .. note::
            keyword argument is intended to help prevent data type overflows.

    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        If the sum was computed over the entire array, a zero-dimensional array
        containing the sum; otherwise, an array containing the sums. The returned array
        must have a data type as described by the ``dtype`` parameter above.

    Examples
    --------
    >>> x = ivy.array([0.41, 0.89])
    >>> y = ivy.sum(x)
    >>> print(y)
    ivy.array(1.3)

    """
    return _cur_framework(x).sum(x, axis, dtype, keepdims, out=out)


def std(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Calculates the standard deviation of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the standard deviation.

    -   If ``N`` is ``0``, the standard deviation is ``0`` (i.e., the empty standard
        deviation).
    -   If ``x_i`` is ``NaN``, the standard deviation is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type

    axis
        axis or axes along which standard deviations must be computed. By default, the
        standard deviation must be computed over the entire array. If a tuple of
        integers, standard deviations must be computed over multiple axes.
        Default: None.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other than 0
        has the effect of adjusting the divisor during the calculation of the standard
        deviation according to N-c where N corresponds to the total number of elements
        over which the standard deviation is computed and c corresponds to the provided
        degrees of freedom adjustment. When computing the standard deviation of a
        population, setting this parameter to ``0`` is the standard choice (i.e., the
        provided array contains data constituting an entire population). When computing
        the corrected sample standard deviation, setting this parameter to ``1`` is the
        standard choice (i.e., the provided array contains data sampled from a larger
        population; this is commonly referred to as Bessel's correction).
        Default: ``0``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the sum was computed over the entire array, a zero-dimensional array
        containing the standard deviation; otherwise, an array containing the standard
        deviations. The returned array must have a data type as described by the
        ``dtype`` parameter above.

    Examples
    --------
    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.std(x)
    >>> print(y)
    ivy.array(0.8164966)

    """
    return _cur_framework(x).std(x, axis, correction, keepdims, out=out)


# Extra #
# ------#


def einsum(
    equation: str,
    *operands: Union[ivy.Array, ivy.NativeArray],
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Sums the product of the elements of the input operands along dimensions specified
    using a notation based on the Einstein summation convention.

    Parameters
    ----------
    equation
        A str describing the contraction, in the same format as numpy.einsum.
    operands
        seq of arrays, the inputs to contract (each one an ivy.Array), whose shapes
        should be consistent with equation.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array with sums computed.

    Examples
    --------
    The following gives us the sum of the diagonal elements:

    >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> y = ivy.einsum('ii', x)
    >>> print(y)
    ivy.array(12)

    Or we can use einsum to sum columns:

    >>> z = ivy.einsum('ij -> j', x)
    >>> print(z)
    ivy.array([9, 12, 15])
    
    Imagine that we have two multi-dimensional arrays, A and B. Now let's suppose we want to...

    - multiply A with B in a particular way to create new array of products; and then maybe
    - sum this new array along particular axes; and then maybe
    - transpose the axes of the new array in a particular order.
    
    There's a good chance that einsum will help us do this faster and more memory-efficiently 
    than combinations of the NumPy functions like multiply, sum and transpose will allow.
    
    >>> A = ivy.array([0, 1, 2])
    >>> B = ivy.array([[ 0,  1,  2,  3],
    ...               [ 4,  5,  6,  7],
    ...               [ 8,  9, 10, 11]])
    >>> ivy.einsum('i,ij->i', A, B)
    ivy.array([ 0, 22, 76])
    
    Now lets see a slightly bigger example:
    
    >>> A = ivy.array([[1, 1, 1],
    ...                [2, 2, 2],
    ...                [5, 5, 5]])
    >>> B = ivy.array([[0, 1, 0],
    ...                [1, 1, 0],
    ...                [1, 1, 1]])
    >>> ivy.einsum('ij,jk->ik', A, B)
    ivy.array([[ 2,  3,  1],
                [ 4,  6,  2],
                [10, 15,  5]])
                
    
    Let A and B be two 1D arrays with the same length. For example, 
    
    >>> A = ivy.arange(10)
    >>> B = ivy.arange(5, 15)
    
    The sum of A can be written:
    >>> ivy.einsum('i->', A)
    ivy.array(45, dtype=int32)
    
    Element-wise multiplication, A * B, can be written:
    >>> ivy.einsum('i,i->i', A, B)
    ivy.array([  0,   6,  14,  24,  36,  50,  66,  84, 104, 126], dtype=int32)
    
    The inner product or dot product can be written:
    >>> ivy.einsum('i,i->', A, B) # or just use 'i,i'
    ivy.array(510, dtype=int32)
    
    The outer product can be written:
    >>> ivy.einsum('i,j->ij', A, B)
    ivy.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                [  5,   6,   7,   8,   9,  10,  11,  12,  13,  14],
                [ 10,  12,  14,  16,  18,  20,  22,  24,  26,  28],
                [ 15,  18,  21,  24,  27,  30,  33,  36,  39,  42],
                [ 20,  24,  28,  32,  36,  40,  44,  48,  52,  56],
                [ 25,  30,  35,  40,  45,  50,  55,  60,  65,  70],
                [ 30,  36,  42,  48,  54,  60,  66,  72,  78,  84],
                [ 35,  42,  49,  56,  63,  70,  77,  84,  91,  98],
                [ 40,  48,  56,  64,  72,  80,  88,  96, 104, 112],
                [ 45,  54,  63,  72,  81,  90,  99, 108, 117, 126]], dtype=int32)
    
    

    """
    return _cur_framework(operands[0]).einsum(equation, *operands, out=out)
