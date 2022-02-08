"""
Collection of math Ivy functions.
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def sin(x, f=None):
    """
    Computes trigonometric sine element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The sine of x element-wise.
    """
    return _cur_framework(x, f=f).sin(x)


def cos(x, f=None):
    """
    Computes trigonometric cosine element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The cosine of x element-wise.
    """
    return _cur_framework(x, f=f).cos(x)


def tan(x, f=None):
    """
    Computes tangent element-wise.
    Equivalent to f.sin(x)/f.cos(x) element-wise.

    :param x: Input array, in radians (2*pi radian equals 360 degrees).
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The tangent of x element-wise.
    """
    return _cur_framework(x, f=f).tan(x)


def asin(x, f=None):
    """
    Computes inverse sine element-wise.

    :param x: y-coordinate on the unit circle.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2].
    """
    return _cur_framework(x, f=f).asin(x)


def acos(x, f=None):
    """
    Computes trigonometric inverse cosine element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    :param x: x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi].
    """
    return _cur_framework(x, f=f).acos(x)


def atan(x, f=None):
    """
    Computes trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if y = tan(x) then x = arctan(y).

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Out has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2).
    """
    return _cur_framework(x, f=f).atan(x)


def atan2(x1, x2, f=None):
    """
    Computes element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    :param x1: y-coordinates.
    :type x1: array
    :param x2: x-coordinates. If x1.shape != x2.shape, they must be broadcastable to a common shape
                    (which becomes the shape of the output).
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Array of angles in radians, in the range [-pi, pi].
    """
    return _cur_framework(x1, f=f).atan2(x1, x2)


def sinh(x, f=None):
    """
    Returns a new array with the hyperbolic sine of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the hyperbolic sine of the elements of x.
    """
    return _cur_framework(x, f=f).sinh(x)


def cosh(x, f=None):
    """
    Returns a new array with the hyperbolic cosine of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the hyperbolic cosine of the elements of x.
    """
    return _cur_framework(x, f=f).cosh(x)


def tanh(x, f=None):
    """
    Returns a new array with the hyperbolic tangent of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x, f=f).tanh(x)


def asinh(x, f=None):
    """
    Returns a new array with the inverse hyperbolic sine of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the inverse hyperbolic sine of the elements of x.
    """
    return _cur_framework(x, f=f).asinh(x)


def acosh(x, f=None):
    """
    Returns a new array with the inverse hyperbolic cosine of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the inverse hyperbolic cosine of the elements of x.
    """
    return _cur_framework(x, f=f).acosh(x)


def atanh(x, f=None):
    """
    Returns a new array with the inverse hyperbolic tangent of the elements of x.

    :param x: Input array.
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array with the inverse hyperbolic tangent of the elements of x.
    """
    return _cur_framework(x, f=f).atanh(x)


def log(x, f=None):
    """
    Computes natural logarithm of x element-wise.

    :param x: Value to compute log for.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The natural logarithm of each element of x.
    """
    return _cur_framework(x, f=f).log(x)


def exp(x, f=None):
    """
    Computes exponential of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The exponential of each element of x.
    """
    return _cur_framework(x, f=f).exp(x)


def erf(x, f=None):
    """
    Computes the Gauss error function of x element-wise.

    :param x: Value to compute exponential for.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The Gauss error function of x.
    """
    return _cur_framework(x, f=f).erf(x)
