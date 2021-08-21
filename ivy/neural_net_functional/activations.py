"""
Collection of Ivy activation functions.
"""

# local
from ivy.framework_handler import current_framework as _cur_framework


def relu(x, f=None):
    """
    Applies the rectified linear unit function element-wise.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with relu applied element-wise.
    """
    return _cur_framework(x, f=f).relu(x)


def leaky_relu(x, alpha=0.2, f=None):
    """
    Applies the leaky rectified linear unit function element-wise.

    :param x: Input array.
    :type x: array
    :param alpha: Negative slope for ReLU
    :type alpha: float
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with leaky relu applied element-wise.
    """
    return _cur_framework(x, f=f).leaky_relu(x, alpha)


def gelu(x, approximate=True, f=None):
    """
    Applies the Gaussian error linear unit (GELU) activation function.

    :param x: Input array.
    :type x: array
    :param approximate: Whether to approximate, default is True.
    :type approximate: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with leaky relu applied element-wise.
    """
    return _cur_framework(x, f=f).gelu(x, approximate)


def tanh(x, f=None):
    """
    Applies the tangent hyperbolic function element-wise.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with tanh applied element-wise.
    """
    return _cur_framework(x, f=f).tanh(x)


def sigmoid(x, f=None):
    """
    Applies the sigmoid function element-wise.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with sigmoid applied element-wise.
    """
    return _cur_framework(x, f=f).sigmoid(x)


def softmax(x, axis=-1, f=None):
    """
    Applies the softmax function element-wise.

    :param x: Input array.
    :type x: array
    :param axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
    :type axis: int, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with softmax applied element-wise.
    """
    return _cur_framework(x, f=f).softmax(x, axis)


def softplus(x, f=None):
    """
    Applies the softplus function element-wise.

    :param x: Input array.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The input array with softplus applied element-wise.
    """
    return _cur_framework(x, f=f).softplus(x)
