from ._types import Optional, Tuple, Union, array, dtype

def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    """
    Calculates the maximum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the maximum value is zero, the maximum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if ``x`` is a floating-point input array, return ``NaN``), or return the minimum possible value for the input array ``x`` data type (e.g., if ``x`` is a floating-point array, return ``-infinity``).

    **Special Cases**

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which maximum values must be computed. By default, the maximum value must be computed over the entire array. If a tuple of integers, maximum values must be computed over multiple axes. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the maximum value was computed over the entire array, a zero-dimensional array containing the maximum value; otherwise, a non-zero-dimensional array containing the maximum values. The returned array must have the same data type as ``x``.
    """

def mean(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
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

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    """
    Calculates the minimum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the minimum value is zero, the minimum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if ``x`` is a floating-point input array, return ``NaN``), or return the maximum possible value for the input array ``x`` data type (e.g., if ``x`` is a floating-point array, return ``+infinity``).

    **Special Cases**

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the minimum value is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which minimum values must be computed. By default, the minimum value must be computed over the entire array. If a tuple of integers, minimum values must be computed over multiple axes. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the minimum value was computed over the entire array, a zero-dimensional array containing the minimum value; otherwise, a non-zero-dimensional array containing the minimum values. The returned array must have the same data type as ``x``.
    """

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[dtype] = None, keepdims: bool = False) -> array:
    """
    Calculates the product of input array ``x`` elements.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the product.

    -   If ``N`` is ``0``, the product is `1` (i.e., the empty product).

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the product is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which products must be computed. By default, the product must be computed over the entire array. If a tuple of integers, products must be computed over multiple axes. Default: ``None``.
    dtype: Optional[dtype]
        data type of the returned array. If ``None``,

        -   if the default data type corresponding to the data type "kind" (integer or floating-point) of ``x`` has a smaller range of values than the data type of ``x`` (e.g., ``x`` has data type ``int64`` and the default data type is ``int32``, or ``x`` has data type ``uint64`` and the default data type is ``int64``), the returned array must have the same data type as ``x``.
        -   if ``x`` has a floating-point data type, the returned array must have the default floating-point data type.
        -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
        -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

        If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the product. Default: ``None``.

        .. note::
           This keyword argument is intended to help prevent data type overflows.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, a non-zero-dimensional array containing the products. The returned array must have a data type as described by the ``dtype`` parameter above.
    """

def std(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    """
    Calculates the standard deviation of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the standard deviation.

    -   If ``N - correction`` is less than or equal to ``0``, the standard deviation is ``NaN``.
    -   If ``x_i`` is ``NaN``, the standard deviation is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which standard deviations must be computed. By default, the standard deviation must be computed over the entire array. If a tuple of integers, standard deviations must be computed over multiple axes. Default: ``None``.
    correction: Union[int, float]
        degrees of freedom adjustment. Setting this parameter to a value other than ``0`` has the effect of adjusting the divisor during the calculation of the standard deviation according to ``N-c`` where ``N`` corresponds to the total number of elements over which the standard deviation is computed and ``c`` corresponds to the provided degrees of freedom adjustment. When computing the standard deviation of a population, setting this parameter to ``0`` is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the corrected sample standard deviation, setting this parameter to ``1`` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel's correction). Default: ``0``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the standard deviation was computed over the entire array, a zero-dimensional array containing the standard deviation; otherwise, a non-zero-dimensional array containing the standard deviations. The returned array must have the same data type as ``x``.

        .. note::
           While this specification recommends that this function only accept input arrays having a floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default floating-point data type.
    """

def sum(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[dtype] = None, keepdims: bool = False) -> array:
    """
    Calculates the sum of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the sum.

    -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which sums must be computed. By default, the sum must be computed over the entire array. If a tuple of integers, sums must be computed over multiple axes. Default: ``None``.
    dtype: Optional[dtype]
        data type of the returned array. If ``None``,

        -   if the default data type corresponding to the data type "kind" (integer or floating-point) of ``x`` has a smaller range of values than the data type of ``x`` (e.g., ``x`` has data type ``int64`` and the default data type is ``int32``, or ``x`` has data type ``uint64`` and the default data type is ``int64``), the returned array must have the same data type as ``x``.
        -   if ``x`` has a floating-point data type, the returned array must have the default floating-point data type.
        -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
        -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

        If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum. Default: ``None``.

        .. note::
           keyword argument is intended to help prevent data type overflows.

    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the sum was computed over the entire array, a zero-dimensional array containing the sum; otherwise, an array containing the sums. The returned array must have a data type as described by the ``dtype`` parameter above.
    """

def var(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    """
    Calculates the variance of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the variance.

    -   If ``N - correction`` is less than or equal to ``0``, the variance is ``NaN``.
    -   If ``x_i`` is ``NaN``, the variance is ``NaN`` (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis or axes along which variances must be computed. By default, the variance must be computed over the entire array. If a tuple of integers, variances must be computed over multiple axes. Default: ``None``.
    correction: Union[int, float]
        degrees of freedom adjustment. Setting this parameter to a value other than ``0`` has the effect of adjusting the divisor during the calculation of the variance according to ``N-c`` where ``N`` corresponds to the total number of elements over which the variance is computed and ``c`` corresponds to the provided degrees of freedom adjustment. When computing the variance of a population, setting this parameter to ``0`` is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the unbiased sample variance, setting this parameter to ``1`` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel's correction). Default: ``0``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the variance was computed over the entire array, a zero-dimensional array containing the variance; otherwise, a non-zero-dimensional array containing the variances. The returned array must have the same data type as ``x``.


    .. note::
       While this specification recommends that this function only accept input arrays having a floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default floating-point data type.
    """

__all__ = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']
