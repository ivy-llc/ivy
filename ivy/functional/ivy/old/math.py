"""
Collection of math Ivy functions.
"""

# local
from ivy.framework_handler import current_framework as _cur_framework



def tan(x):
    """
    Computes tangent element-wise.
    Equivalent to f.sin(x)/f.cos(x) element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array
    :return: The tangent of x element-wise.
    """
    return _cur_framework(x).tan(x)


def asin(x):
    """
    Computes inverse sine element-wise.

    :param x: y-coordinate on the unit circle.
    :type x: array
    :return: The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2].
    """
    return _cur_framework(x).asin(x)


def acos(x):
    """
    Computes trigonometric inverse cosine element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    :param x: x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    :type x: array
    :return: The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi].
    """
    return _cur_framework(x).acos(x)


def atan(x):
    """
    Computes trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if y = tan(x) then x = arctan(y).

    :param x: Input array.
    :type x: array
    :return: Out has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2).
    """
    return _cur_framework(x).atan(x)


def atan2(x1, x2):
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


def cosh(x):
    """
    Returns a new array with the hyperbolic cosine of the elements of x.

    :param x: Input array.
    :return: A new array with the hyperbolic cosine of the elements of x.
    """
    return _cur_framework(x).cosh(x)


def tanh(x):
    """
    Returns a new array with the hyperbolic tangent of the elements of x.

    :param x: Input array.
    :return: A new array with the hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x).tanh(x)


def atanh(x):
    """
    Returns a new array with the inverse hyperbolic tangent of the elements of x.

    :param x: Input array.
    :return: A new array with the inverse hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x).atanh(x)


def log(x):
    """
    Computes natural logarithm of x element-wise.

    :param x: Value to compute log for.
    :type x: array
    :return: The natural logarithm of each element of x.
    """
    return _cur_framework(x).log(x)


def exp(x):
    """
    Computes exponential of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :return: The exponential of each element of x.
    """
    return _cur_framework(x).exp(x)


def erf(x):
    """
    Computes the Gauss error function of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :return: The Gauss error function of x.
    """
    return _cur_framework(x).erf(x)


def divide(x1, x2):
    """
    Calculates the division for each element x1_i of the input array x1 with the respective element x2_i of the
    input array x2.

    :param x1: dividend input array. Should have a numeric data type.
    :param x2: divisor input array. Must be compatible with x1 (see Broadcasting). Should have a numeric data type.
    :return: an array containing the element-wise results. The returned array must have a floating-point data type
             determined by Type Promotion Rules.
    """
    return x1 / x2

