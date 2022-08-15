# global
from typing import Union, Tuple, Optional, Sequence

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def max(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculates the maximum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the maximum value is zero, the
       maximum value is implementation-defined. Specification-compliant libraries may
       choose to raise an error, return a sentinel value (e.g., if ``x`` is a
       floating-point input array, return ``NaN``), or return the minimum possible
       value for the input array ``x`` data type (e.g., if ``x`` is a floating-point
       array, return ``-infinity``).

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

    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.max.html>`_  # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> z = x.max()
    >>> print(z)
    ivy.array(3)

    >>> x = ivy.array([0, 1, 2])
    >>> z = ivy.array([0,0,0])
    >>> y = ivy.max(x, out=z)
    >>> print(z)
    ivy.array(2)

    >>> x = ivy.array([[0, 1, 2], [4, 6, 10]])
    >>> y = ivy.max(x, 0, True)
    >>> print(y)
    ivy.array([[4, 6, 10]])

    >>> x = ivy.native_array([[0, 1, 2], [4, 6, 10]])
    >>> y = ivy.max(x)
    >>> print(y)
    ivy.array(10)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.max(x)
    >>> print(y)
    {
        a: ivy.array(2.),
        b: ivy.array(5.)
    }

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                          b=ivy.array([2, 3, 4]))
    >>> z = x.max()
    >>> print(z)
    {
        a: ivy.array(3),
        b: ivy.array(4)
    }
    """
    return current_backend(x).max(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def mean(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    *,
    out: Optional[ivy.Array] = None,
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
           While mixed data type promotion is implementation-defined, if the input
           array ``x`` has an integer data type, the returned array must have the
           default floating-point data type.

    """
    return current_backend(x).mean(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def min(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Union[int, Tuple[int]] = None,
    keepdims: bool = False,
    *,
    out: Optional[ivy.Array] = None,
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
    return current_backend.min(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def prod(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
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

    >>> x = ivy.array([1, 2, 3])
    >>> z = ivy.prod(x)
    >>> print(z)
    ivy.array(6)

    >>> x = ivy.array([1, 0, 3])
    >>> z = ivy.prod(x)
    >>> print(z)
    ivy.array(0)

    """
    return current_backend(x).prod(
        x, axis=axis, dtype=dtype, keepdims=keepdims, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def std(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    *,
    out: Optional[ivy.Array] = None,
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
    return current_backend(x).std(x, axis, correction, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def sum(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
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
    return current_backend(x).sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def var(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculates the variance of the input array x.

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

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = ivy.var(x)
    >>> print(y)
    ivy.array(0.07472222)

    >>> x = ivy.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = ivy.zeros(6)
    >>> ivy.var(x, out=y)
    >>> print(y)
    ivy.array(0.07472222)

    >>> x = ivy.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> ivy.var(x, out=x)
    >>> print(x)
    ivy.array(0.07472222)

    With :code:`ivy.native_array` input:

    >>> x = ivy.native_array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = ivy.var(x)
    >>> print(y)
    ivy.array(0.07472222)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.1, 0.2, 0.9]), \
                          b=ivy.array([0.7, 0.1, 0.9]))
    >>> y = ivy.var(x)
    >>> print(y)
    {
        a: ivy.array(0.12666667),
        b: ivy.array(0.11555555)
    }

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/
    signatures.elementwise_functions.tan.html>`_ in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> y = ivy.var(x)
    >>> print(y)
    ivy.array(6.6666665)

    >>> x = ivy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> y = ivy.array(0.0)
    >>> ivy.var(x, out=y)
    >>> print(y)
    ivy.array(6.6666665)

    >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    >>> y = ivy.array(0.0)
    >>> ivy.var(x, out=y)
    >>> print(y)
    ivy.array(6.6666665)

    >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0],[6.0, 7.0, 8.0]])
    >>> y = ivy.zeros(3)
    >>> ivy.var(x, axis=1, out=y)
    >>> print(y)
    ivy.array([0.667, 0.667, 0.667])

    >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    >>> y = ivy.zeros(3)
    >>> ivy.var(x, axis=0, out=y)
    >>> print(y)
    ivy.array([6., 6., 6.])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1.0, 2.0, 2.0, 3.0])
    >>> y = ivy.var(x)
    >>> print(y)
    ivy.array(0.5)

    >>> x = ivy.native_array([1.0, 2.0, 2.0, 3.0])
    >>> y = ivy.array(0.0)
    >>> ivy.var(x, out=y)
    >>> print(y)
    ivy.array(0.5)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]), b=ivy.array([3.0, 4.0, 5.0]))
    >>> y = ivy.var(x)
    >>> print(y)
    {
        a: ivy.array(0.6666667),
        b: ivy.array(0.6666667)
    }

    >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]), b=ivy.array([3.0, 4.0, 5.0]))
    >>> y = ivy.Container.static_var(x)
    >>> print(y)
    {
        a: ivy.array(0.6666667),
        b: ivy.array(0.6666667)
    }

    Instance Method Examples
    ------------------------
    Using :code:`ivy.Array` instance method:
    
    >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0],[6.0, 7.0, 8.0]])
    >>> y = x.var()
    >>> print(y)
    ivy.array(6.6666665)

    >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0],[6.0, 7.0, 8.0]])
    >>> y = x.var(axis=0)
    >>> print(y)
    ivy.array([6., 6., 6.])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]), b=ivy.array([3.0, 4.0, 5.0]))
    >>> y = x.var()
    >>> print(y)
    {
        a: ivy.array(0.6666667),
        b: ivy.array(0.6666667)
    }

    """
    return current_backend(x).var(x, axis, correction, keepdims, out=out)


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def einsum(
    equation: str,
    *operands: Union[ivy.Array, ivy.NativeArray],
    out: Optional[ivy.Array] = None,
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

    Functional Examples
    -------------------

    With :code: 'ivy.Array' input:

    >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> y = ivy.einsum('ii', x)
    >>> print(y)
    ivy.array(12)

    >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> z = ivy.einsum('ij -> j', x)
    >>> print(z)
    ivy.array([ 9, 12, 15])

    >>> A = ivy.array([0, 1, 2])
    >>> B = ivy.array([[ 0,  1,  2,  3],\
                       [ 4,  5,  6,  7],\
                       [ 8,  9, 10, 11]])
    >>> C = ivy.einsum('i,ij->i', A, B)
    >>> print(C)
    ivy.array([ 0, 22, 76])

    >>> A = ivy.array([[1, 1, 1],\
                       [2, 2, 2],\
                       [5, 5, 5]])
    >>> B = ivy.array([[0, 1, 0],\
                       [1, 1, 0],\
                       [1, 1, 1]])
    >>> C = ivy.einsum('ij,jk->ik', A, B)
    >>> print(C)
    ivy.array([[ 2,  3,  1],
           [ 4,  6,  2],
           [10, 15,  5]])

    >>> A = ivy.arange(10)
    >>> B = ivy.arange(5, 15)
    >>> C = ivy.einsum('i->', A)
    >>> print(C)
    ivy.array(45)

    >>> A = ivy.arange(10)
    >>> B = ivy.arange(5, 15)
    >>> C = ivy.einsum('i,i->i', A, B)
    >>> print(C)
    ivy.array([  0,   6,  14,  24,  36,  50,  66,  84, 104, 126])

    >>> A = ivy.arange(10)
    >>> B = ivy.arange(5, 15)
    >>> C = ivy.einsum('i,i->', A, B) # or just use 'i,i'
    >>> print(C)
    ivy.array(510)

    >>> A = ivy.arange(10)
    >>> B = ivy.arange(5, 15)
    >>> C = ivy.einsum('i,j->ij', A, B)
    >>> print(C)
    ivy.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  5,   6,   7,   8,   9,  10,  11,  12,  13,  14],
           [ 10,  12,  14,  16,  18,  20,  22,  24,  26,  28],
           [ 15,  18,  21,  24,  27,  30,  33,  36,  39,  42],
           [ 20,  24,  28,  32,  36,  40,  44,  48,  52,  56],
           [ 25,  30,  35,  40,  45,  50,  55,  60,  65,  70],
           [ 30,  36,  42,  48,  54,  60,  66,  72,  78,  84],
           [ 35,  42,  49,  56,  63,  70,  77,  84,  91,  98],
           [ 40,  48,  56,  64,  72,  80,  88,  96, 104, 112],
           [ 45,  54,  63,  72,  81,  90,  99, 108, 117, 126]])

    With :code:'ivy.NativeArray' input:

    >>> x = ivy.native_array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> y = ivy.einsum('ii', x)
    >>> print(y)
    ivy.array(12)

    With a mix of code: 'ivy.Array' and code: 'ivy.NativeArray' inputs:

    >>> A = ivy.array([0, 1, 2])
    >>> B = ivy.native_array([[ 0, 1, 2, 3],\
                              [ 4, 5, 6, 7],\
                              [ 8, 9, 10, 11]])
    >>> C = ivy.einsum('i,ij->i', A, B)
    >>> print(C)
    ivy.array([ 0, 22, 76])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0, 1, 2])
    >>> y = ivy.Container(a=ivy.array([[ 0,  1,  2,  3],\
                                       [ 4,  5,  6,  7],\
                                       [ 8,  9, 10, 11]]),\
                          b=ivy.array([[ 0,  1,  2],\
                                       [ 4,  5,  6],\
                                       [ 8,  9, 10]]))
    >>> z = ivy.einsum('i,ij->i', x, y)
    >>> print(z)
    {
        a: ivy.array([0, 22, 76]),
        b: ivy.array([0, 15, 54])
    }

    With :code: 'ivy.Container' input:

    >>> x = ivy.Container(a=ivy.array([[0, 1, 0],[1, 1, 0],[1, 1, 1]]),\
                          b=ivy.array([[0, 1, 2],[4, 5, 6],[8, 9, 10]]))
    >>> y = ivy.einsum('ii', x)
    >>> print(y)
    {
        a: ivy.array(2),
        b: ivy.array(15)
    }

    Instance Method Examples
    ------------------------

    Using :code: 'ivy.Array' instance method:

    >>> x = ivy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> y = x.einsum('ii')
    >>> print(y)
    ivy.array(12)

    Using :code: 'ivy.Container' instance method:

    >>> x = ivy.Container(a=ivy.array([[0, 1, 0],[1, 1, 0],[1, 1, 1]]),\
                          b=ivy.array([[0, 1, 2],[4, 5, 6],[8, 9, 10]]))
    >>> y = x.einsum('ii')
    >>> print(y)
    {
        a: ivy.array(2),
        b: ivy.array(15)
    }

    """
    return current_backend(operands[0]).einsum(equation, *operands, out=out)
