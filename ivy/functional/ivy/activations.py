"""
Collection of Ivy activation functions.
"""

# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Extra #
# ------#

def relu(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Applies the rectified linear unit function element-wise.

     Parameters
     ----------
     x:
         input array


    Returns
    -------
    out:
       an array containing the rectified linear unit activation of each element in ``x``.

    Examples:
    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.relu(x)
    >>> print(y)
    [-0.0, 0.0, 1.0]
    """
    return _cur_framework(x).relu(x)


def leaky_relu(x, alpha=0.2):
    """
    Applies the leaky rectified linear unit function element-wise.

    :param x: Input array.
    :type x: array
    :param alpha: Negative slope for ReLU
    :type alpha: float
    :return: The input array with leaky relu applied element-wise.
    """
    return _cur_framework(x).leaky_relu(x, alpha)


def gelu(x, approximate=True):
    """
    Applies the Gaussian error linear unit (GELU) activation function.

    :param x: Input array.
    :type x: array
    :param approximate: Whether to approximate, default is True.
    :type approximate: bool, optional
    :return: The input array with leaky relu applied element-wise.
    """
    return _cur_framework(x).gelu(x, approximate)


def tanh(x):
    """
    Applies the tangent hyperbolic function element-wise.

    :param x: Input array.
    :type x: array
    :return: The input array with tanh applied element-wise.
    """
    return _cur_framework(x).tanh(x)


def sigmoid(x):
    """
    Applies the sigmoid function element-wise.

    :param x: Input array.
    :type x: array
    :return: The input array with sigmoid applied element-wise.
    """
    return _cur_framework(x).sigmoid(x)


def softmax(x, axis=-1):
    """
    Applies the softmax function element-wise.

    :param x: Input array.
    :type x: array
    :param axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
    :type axis: int, optional
    :return: The input array with softmax applied element-wise.
    """
    return _cur_framework(x).softmax(x, axis)


def softplus(x):
    """
    Applies the softplus function element-wise.

    :param x: Input array.
    :type x: array
    :return: The input array with softplus applied element-wise.
    """
    return _cur_framework(x).softplus(x)
