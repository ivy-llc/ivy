# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def expm1(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to ``exp(x)-1``, having domain ``[-infinity, +infinity]`` and codomain ``[-1, +infinity]``, for each element ``x_i`` of the input array ``x``.
    .. note::
       The purpose of this function is to calculate ``exp(x)-1.0`` more accurately when `x` is close to zero. Accordingly, conforming implementations should avoid implementing this function as simply ``exp(x)-1.0``. See FDLIBM, or some other IEEE 754-2019 compliant mathematical library, for a potential reference implementation.
    **Special cases**
    For floating-point operands,
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-1``.
    Parameters
    ----------
    x: array
        input array. Should have a numeric data type.
    Returns
    -------
    out: array
        an array containing the evaluated result for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).expm1(x)
  
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


def log(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the natural (base ``e``) logarithm, having domain ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` of the input array ``x``.
    **Special cases**
    For floating-point operands,
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    Returns
    -------
    out: array
        an array containing the evaluated natural logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x).log(x)


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


def logical_xor(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes the bitwise XOR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.
    Parameters
    ----------
    x1: array
        first input array. Should have an integer or boolean data type.
    x2: array
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.
    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.
    """
    return _cur_framework(x1, x2).logical_xor(x1, x2)


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


def round(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Rounds the values of an array to the nearest integer, element-wise.

    :param x: Input array containing elements to round.
    :type x: array
    :return: An array of the same shape and type as x, with the elements rounded to integers.
    """
    return _cur_framework(x).round(x)


def abs(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Calculates the absolute value for each element ``x_i`` of the input array ``x`` (i.e., the element-wise result has the same magnitude as the respective element in ``x`` but has positive sign).

    .. note::
        For signed integer data types, the absolute value of the minimum representable integer is implementation-dependent.

    **Special Cases**

    For this particular case,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``-0``, the result is ``+0``.
    - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the absolute value of each element in ``x``. The returned array must have the same data type as ``x``.
    """
    return _cur_framework(x).abs(x)


def tan(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes tangent element-wise.
    Equivalent to f.sin(x)/f.cos(x) element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array
    :return: The tangent of x element-wise.
    """
    return _cur_framework(x).tan(x)


def asin(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes inverse sine element-wise.

    :param x: y-coordinate on the unit circle.
    :type x: array
    :return: The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2].
    """
    return _cur_framework(x).asin(x)


def atan(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if y = tan(x) then x = arctan(y).

    :param x: Input array.
    :type x: array
    :return: Out has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2).
    """
    return _cur_framework(x).atan(x)


def atan2(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    :param x1: y-coordinates.
    :type x1: array
    :param x2: x-coordinates. If x1.shape != x2.shape, they must be broadcastable to a common shape
                    (which becomes the shape of the output).
    :type x2: array
    :return: Array of angles in radians, in the range [-pi, pi].
    """
    return _cur_framework(x1).atan2(x1, x2)


def cosh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Returns a new array with the hyperbolic cosine of the elements of x.

    :param x: Input array.
    :return: A new array with the hyperbolic cosine of the elements of x.
    """
    return _cur_framework(x).cosh(x)


def tanh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Returns a new array with the hyperbolic tangent of the elements of x.

    :param x: Input array.
    :return: A new array with the hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x).tanh(x)


def atanh(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Returns a new array with the inverse hyperbolic tangent of the elements of x.

    :param x: Input array.
    :return: A new array with the inverse hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x).atanh(x)


def log(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes natural logarithm of x element-wise.

    :param x: Value to compute log for.
    :type x: array
    :return: The natural logarithm of each element of x.
    """
    return _cur_framework(x).log(x)


def exp(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes exponential of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :return: The exponential of each element of x.
    """
    return _cur_framework(x).exp(x)


def divide(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Calculates the division for each element x1_i of the input array x1 with the respective element x2_i of the
    input array x2.

    :param x1: dividend input array. Should have a numeric data type.
    :param x2: divisor input array. Must be compatible with x1 (see Broadcasting). Should have a numeric data type.
    :return: an array containing the element-wise results. The returned array must have a floating-point data type
             determined by Type Promotion Rules.
    """
    return x1 / x2


def remainder(x1: Union[ivy.Array, ivy.NativeArray],
              x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Returns the remainder of division for each element ``x1_i`` of the input array ``x1``
    and the respective element ``x2_i`` of the input array ``x2``.

    .. note::
        This function is equivalent to the Python modulus operator ``x1_i % x2_i``.
        For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.
        In general, similar to Python’s ``%`` operator, this function is not recommended for floating-point operands as semantics do not follow IEEE 754. That this function is specified to accept floating-point operands is primarily for reasons of backward compatibility.

    **Special Cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
    - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
    - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
    - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
    - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
    - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
    - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x1_i``. (note: this result matches Python behavior.)
    - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x2_i``. (note: this result matches Python behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x2_i``. (note: this results matches Python behavior.)
    - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x1_i``. (note: this result matches Python behavior.)
    - In the remaining cases, the result must match that of the Python ``%`` operator.

    Parameters
    ----------
    x1:
        dividend input array. Should have a numeric data type.
    x2:
        divisor input array. Must be compatible with ``x1`` (see :ref:`Broadcasting`). Should have a numeric data type.

    Returns
    -------
    out:
        an array containing the element-wise results. Each element-wise result must have the same sign as the respective element ``x2_i``. The returned array must have a data type determined by :ref:`Type Promotion Rules`.
    """
    return _cur_framework(x1, x2).remainder(x1, x2)


# Extra #
# ------#


def erf(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Computes the Gauss error function of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :return: The Gauss error function of x.
    """
    return _cur_framework(x).erf(x)
