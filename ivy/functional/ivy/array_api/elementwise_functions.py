# local
import ivy
from typing import Union
from ivy.framework_handler import current_framework as _cur_framework



def bitwise_and(x1: Union[ivy.Array, ivy.NativeArray],
                x2: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes the bitwise AND of the underlying binary representation of each element x1_i of the input array x1 with
    the respective element x2_i of the input array x2.

    :param x1: first input array. Should have an integer or boolean data type.
    :param x2: second input array. Must be compatible with x1 (see Broadcasting). Should have an integer or
               boolean data type.
    :return: an array containing the element-wise results. The returned array must have a data type determined
             by Type Promotion Rules.
    """
    return _cur_framework(x1, x2).bitwise_and(x1, x2)



def ceil(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Returns element-wise smallest integer not less than x.

    :param x: Input array to ceil.
    :return: An array of the same shape and type as x, with the elements ceiled to integers.
    """
    return _cur_framework(x).ceil(x)


def isfinite(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Tests each element x_i of the input array x to determine if finite (i.e., not NaN and not equal to positive
    or negative infinity).

    :param x: input array. Should have a numeric data type.
    :return: an array containing test results. An element out_i is True if x_i is finite and False otherwise.
             The returned array must have a data type of bool.
    """
    return _cur_framework(x).isfinite(x)


def asinh(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the inverse hyperbolic sine, having domain
    ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i`` in the input array ``x``.

    **Special cases**
    For floating-point operands,
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

    :param x: input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point
              data type.
    :return: an array containing the inverse hyperbolic sine of each element in ``x``. The returned array must have a
             floating-point data type determined by type-promotion.
    """
    return _cur_framework(x).asinh(x)


def sqrt(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Calculates the square root, having domain [0, +infinity] and codomain [0, +infinity], for each element x_i of the
    input array x. After rounding, each result must be indistinguishable from the infinitely precise result (as required
     by IEEE 754).

     :param x: input array. Should have a floating-point data type.
     :return: an array containing the square root of each element in x. The returned array must have a floating-point
     data type determined by Type Promotion Rules.
    """
    return _cur_framework(x).sqrt(x)


def cosh(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Returns a new array with the hyperbolic cosine of the elements of x.

    :param x: Input array.
    :return: A new array with the hyperbolic cosine of the elements of x.
    """
    return _cur_framework(x).cosh(x)


def log2(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Calculates an implementation-dependent approximation to the base 2 logarithm.

    :param x: Input array.
    :return: A new array containing the evaluated base 2 logarithm for each element in x.
    """
    return _cur_framework(x).log2(x)


def isnan(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Returns boolean map at locations where the input is not a number (nan).

    :param x: Input array.
    :return: Boolean values for where the values of the array are nan.
    """
    return _cur_framework(x).isnan(x)


def less(x1: Union[ivy.Array, ivy.NativeArray],
         x2: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes the truth value of x1_i < x2_i for each element x1_i of the input array x1 with the respective 
    element x2_i of the input array x2.

    :param x1: Input array.
    :param x2: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :return: an array containing the element-wise results. The returned array must have a data type of bool.
    """

    return _cur_framework(x1).less(x1,x2)



def cos(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes trigonometric cosine element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :return: The cosine of x element-wise.
    """
    return _cur_framework(x).cos(x)


def logical_not(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Computes the truth value of NOT x element-wise.

    :param x: Input array.
    :return: Boolean result of the logical NOT operation applied element-wise to x.
    """
    return _cur_framework(x).logical_not(x)


def divide(x1: Union[ivy.Array, ivy.NativeArray],
           x2: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Calculates the division for each element x1_i of the input array x1
    with the respective element x2_i of the input array x2.

    **special cases**
    For floating-point operands,
    - If either x1_i or x2_i is NaN, the result is NaN.
    - If x1_i is either +infinity or -infinity and x2_i is either +infinity or -infinity, the result is NaN.
    - If x1_i is either +0 or -0 and x2_i is either +0 or -0, the result is NaN.
    - If x1_i is +0 and x2_i is greater than 0, the result is +0.
    - If x1_i is -0 and x2_i is greater than 0, the result is -0.
    - If x1_i is +0 and x2_i is less than 0, the result is -0.
    - If x1_i is -0 and x2_i is less than 0, the result is +0.
    - If x1_i is greater than 0 and x2_i is +0, the result is +infinity.
    - If x1_i is greater than 0 and x2_i is -0, the result is -infinity.
    - If x1_i is less than 0 and x2_i is +0, the result is -infinity.
    - If x1_i is less than 0 and x2_i is -0, the result is +infinity.
    - If x1_i is +infinity and x2_i is a positive (i.e., greater than 0) finite number, the result is +infinity.
    - If x1_i is +infinity and x2_i is a negative (i.e., less than 0) finite number, the result is -infinity.
    - If x1_i is -infinity and x2_i is a positive (i.e., greater than 0) finite number, the result is -infinity.
    - If x1_i is -infinity and x2_i is a negative (i.e., less than 0) finite number, the result is +infinity.
    - If x1_i is a positive (i.e., greater than 0) finite number and x2_i is +infinity, the result is +0.
    - If x1_i is a positive (i.e., greater than 0) finite number and x2_i is -infinity, the result is -0.
    - If x1_i is a negative (i.e., less than 0) finite number and x2_i is +infinity, the result is -0.
    - If x1_i is a negative (i.e., less than 0) finite number and x2_i is -infinity, the result is +0.
    - If x1_i and x2_i have the same mathematical sign and are both nonzero finite numbers,
      the result has a positive mathematical sign.
    - If x1_i and x2_i have different mathematical signs and are both nonzero finite numbers,
      the result has a negative mathematical sign.
    - If neither -infinity, +0, -0, nor NaN is involved, the quotient must be computed
      and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode.
    - If the magnitude is too large to represent,
      the operation overflows and the result is an infinity of appropriate mathematical sign.
    - If the magnitude is too small to represent,
      the operation under-flows and the result is a zero of appropriate mathematical sign.

    :params x1 (array): dividend input array. Should have a numeric data type.
            x2 (array): divisor input array. Must be compatible with x1 (see Broadcasting).
            Should have a numeric data type.

    :return: an array containing the element-wise results.
    The returned array must have a floating-point data type determined by Type Promotion Rules.
    """

    return _cur_framework(x1, x2).divide(x1, x2)
