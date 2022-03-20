# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def bitwise_invert(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Inverts (flips) each bit for each element x_i of the input array x.

    Parameters
    ----------
    x:
        input array. Should have an integer or boolean data type.

    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have the same data type as x.
    """
    return _cur_framework(x).bitwise_invert(x)


def bitwise_and(x1: Union[ivy.Array, ivy.NativeArray],
                x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Computes the bitwise AND of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1:
        first input array. Should have an integer or boolean data type.
    x2:
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x1, x2).bitwise_and(x1, x2)


def ceil(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the smallest (i.e., closest to ``-infinity``) integer-valued number that is not less than ``x_i``.

    **Special cases**

    - If ``x_i`` is already integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.
    """
    return _cur_framework(x).ceil(x)


def floor(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the greatest (i.e., closest to ``+infinity``) integer-valued number that is not greater than ``x_i``.

    **Special cases**

    - If ``x_i`` is already integer-valued, the result is ``x_i``.

    For floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.
    """
    return _cur_framework(x).floor(x)


def isfinite(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine if finite (i.e., not ``NaN`` and not equal to positive or negative infinity).

    Parameters
    ----------
    x:
       input array. Should have a numeric data type.

    Returns
    -------
    out:
       an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is finite and ``False`` otherwise. The returned array must have a data type of ``bool``.
    """
    return _cur_framework(x).isfinite(x)

  
def asin(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation of the principal value of the inverse sine, having domain ``[-1, +1]`` and codomain ``[-π/2, +π/2]`` for each element ``x_i`` of the input array ``x``. Each element-wise result is expressed in radians.
    
    **Special cases**
    
    For floating-point operands,
    
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is greater than ``1``, the result is ``NaN``.
    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    
    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
        
    Returns
    -------
    out: array
        an array containing the inverse sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).asin(x)

  
def isinf(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Tests each element x_i of the input array x to determine if equal to positive or negative infinity.
    Parameters
    ----------
    x:
        input array. Should have a numeric data type.
    Returns
    -------
    out:
        an array containing test results. An element out_i is True if x_i is either positive or negative infinity and False otherwise. The returned array must have a data type of bool.
    """
    return _cur_framework(x).isinf(x)


def greater_equal(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Computes the truth value of x1_i >= x2_i for each element x1_i of the input array x1 with the respective
    element x2_i of the input array x2.

    :param x1: first input array. May have any data type.
    :param x2: second input array. Must be compatible with x1 (with Broadcasting). May have any data type.
    :return: an array containing the element-wise results. The returned array must have a data type of bool.
    """
    return _cur_framework(x1, x2).greater_equal(x1, x2)


def less_equal(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes the truth value of x1_i <= x2_i for each element x1_i of the input array x1 with the respective
    element x2_i of the input array x2.

    :param x1: first input array. May have any data type.
    :param x2: second input array. Must be compatible with x1 (with Broadcasting). May have any data type.
    :return: an array containing the element-wise results. The returned array must have a data type of bool.
    """
    return _cur_framework(x1, x2).less_equal(x1, x2)


def asinh(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the inverse hyperbolic sine, having domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` in the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

    Parameters
    ----------
    x:
        input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the inverse hyperbolic sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).asinh(x)


def sqrt(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates the square root, having domain ``[0, +infinity]`` and codomain ``[0, +infinity]``, for each element ``x_i`` of the input array ``x``. After rounding, each result must be indistinguishable from the infinitely precise result (as required by IEEE 754).
    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x:
        input array. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the square root of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).sqrt(x)


def cosh(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the hyperbolic cosine, having domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` in the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``1``.
    - If ``x_i`` is ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x:
        input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the hyperbolic cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

    return _cur_framework(x).cosh(x)


def log2(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the base ``2`` logarithm, having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x:
        input array. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the evaluated base ``2`` logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).log2(x)


def log10(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the base ``10`` logarithm, having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` of the input array ``x``.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x:
        input array. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the evaluated base ``10`` logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).log10(x)


def log1p(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to log(1+x), where log refers to the natural (base e)
    logarithm.

    Parameters
    ----------
    x:
        input array.

    Returns
    -------
    out:
        a new array containing the evaluated result for each element in x.
    """
    return _cur_framework(x).log1p(x)


def isnan(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.

    Returns
    -------
    out:
        an array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is ``NaN`` and ``False`` otherwise. The returned array should have a data type of ``bool``.
    """
    return _cur_framework(x).isnan(x)


def less(x1: Union[ivy.Array, ivy.NativeArray],
         x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1:
        first input array. Should have a numeric data type.
    x2:
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have a data type of ``bool``.
    """
    return _cur_framework(x1).less(x1,x2)


def cos(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the cosine, having domain ``(-infinity, +infinity)`` and codomain ``[-1, +1]``, for each element ``x_i`` of the input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``1``.
    - If ``x_i`` is ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity``, the result is ``NaN``.
    - If ``x_i`` is ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x:
        input array whose elements are each expressed in radians. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
        """
    return _cur_framework(x).cos(x)


def logical_not(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Computes the logical NOT for each element ``x_i`` of the input array ``x``.

    .. note::
       While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having numeric data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

    Parameters
    ----------
    x:
        input array. Should have a boolean data type.

    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have a data type of ``bool``.
    """
    return _cur_framework(x).logical_not(x)


def acos(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation of the principal value of the inverse cosine, having domain [-1, +1] and codomain [+0, +π], for each element x_i of the input array x. Each element-wise result is expressed in radians.

    **Special cases**

    For floating-point operands,

    - If x_i is NaN, the result is NaN.
    - If x_i is greater than 1, the result is NaN.
    - If x_i is less than -1, the result is NaN.
    - If x_i is 1, the result is +0.
    
    Parameters
    ----------
    x:
        input array. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the inverse cosine of each element in x. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).acos(x)


def logical_or(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having numeric data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

    Parameters
    ----------
    x1:
        first input array. Should have a boolean data type.
    x2:
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a boolean data type.

    Returns
    -------
    out:
        out (array) – an array containing the element-wise results. The returned array must have a data type of ``bool``.
    """
    return _cur_framework(x1, x2).logical_or(x1, x2)


def logical_and(x1: ivy.Array, x2: ivy.Array)\
       -> ivy.Array:
    """
    Computes the logical AND for each element x1_i of the input array x1 with the respective
    element x2_i of the input array x2.

    Parameters
    ----------
    x1:
        first input array. Should have a boolean data type.
    x2:
        second input array. Must be compatible with x1.
        Should have a boolean data type.
    Returns
    -------
    out:
        out (array) – an array containing the element-wise results.
        The returned array must have a data type of bool.
    """
    return _cur_framework(x1, x2).logical_and(x1, x2)


def acosh(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the inverse hyperbolic cosine, having domain ``[+1, +infinity]`` and codomain ``[+0, +infinity]``, for each element ``x_i`` of the input array ``x``.
    
    **Special cases**

    For floating-point operands,
    
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``1``, the result is ``NaN``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    
    Parameters
    ----------
    x:
        input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the inverse hyperbolic cosine of each element in x. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).acosh(x)
    

def sin(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the sine, having domain ``(-infinity, +infinity)`` and codomain ``[-1, +1]``, for each element ``x_i`` of the input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x:
        input array whose elements are each expressed in radians. Should have a floating-point data type.

    Returns
    -------
    out:
        an array containing the sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).sin(x)


def negative(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes the numerical negative of each element
    
    :param x: Input array
    :return: an array containing the evaluated result for each element in x 
    """
    return _cur_framework(x).negative(x)


def not_equal(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1:
        first input array. Should have a numeric data type.
    x2:
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have a data type of ``bool``.
    """
    return _cur_framework(x1, x2).not_equal(x1, x2)


def tanh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the hyperbolic tangent,
    having domain [-infinity, +infinity] and codomain [-1, +1], for each element x_i of the input array x.

    :param x: input array whose elements each represent a hyperbolic angle. Should have a floating-point
            data type.
    :return: an array containing the hyperbolic tangent of each element in x. The returned array must
            have a floating-point data type
    """
    return _cur_framework(x).tanh(x)


def bitwise_or(x1: Union[ivy.Array, ivy.NativeArray],
                x2: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes the bitwise OR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.
    
    Parameters
    ----------
    x1:
        first input array. Should have an integer or boolean data type.
    x2:
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.
    
    Returns
    -------
    out:
        an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x1, x2).bitwise_or(x1, x2)


def sinh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the hyperbolic sine, having domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` of the input array ``x``.
    
    **Special cases**
    
    For floating-point operands,
    
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
    
    Parameters
    ----------
    x: 
        input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.
    
    Returns
    -------
    out:
        an array containing the hyperbolic sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).sinh(x)


def positive(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Returns a new array with the positive value of each element in x.

    :param x: Input array.
    :return: A new array with the positive value of each element in x.
    """
    return _cur_framework(x).positive(x)

    
def square(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    each element x_i of the input array x.
    :param x: Input array.
    :return: an array containing the evaluated result for each element in x.
    """
    return _cur_framework(x).square(x)


def logaddexp(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    
    """
    Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.
    **Special cases**
    For floating-point operands,
    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is not ``NaN``, the result is ``+infinity``.
    - If ``x1_i`` is not ``NaN`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x1: 
        first input array. Should have a floating-point data type.
    x2: 
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a floating-point data type.
    Returns
    -------
    out: 
        an array containing the element-wise results. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

    return _cur_framework(x1, x2).logaddexp(x1, x2)

# Extra #
# ------#
def round(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Rounds the values of an array to the nearest integer, element-wise.

    :param x: Input array containing elements to round.
    :type x: array
    :return: An array of the same shape and type as x, with the elements rounded to integers.
    """
    return _cur_framework(x).round(x)

