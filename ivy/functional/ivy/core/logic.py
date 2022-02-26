"""
Collection of logic Ivy functions.
"""

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def logical_and(x1, x2):
    """
    Computes the truth value of x1 AND x2 element-wise.

    :param x1: Input array 1.
    :type x1: array
    :param x2: Input array 2.
    :type x2: array
    :return: Boolean result of the logical AND operation applied element-wise to x1 and x2.
    """
    return _cur_framework(x1).logical_and(x1, x2)


def logical_or(x1, x2):
    """
    Computes the truth value of x1 OR x2 element-wise.

    :param x1: Input array 1.
    :type x1: array
    :param x2: Input array 2.
    :type x2: array
    :return: Boolean result of the logical OR operation applied element-wise x1 and x2.
    """
    return _cur_framework(x1).logical_or(x1, x2)
