"""
Collection of Ivy loss functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


# noinspection PyUnresolvedReferences
def binary_cross_entropy(x, y, epsilon=1e-7, f=None):
    """
    Computes the cross-entropy loss between true labels and predicted labels.

    :param x: Predicted labels
    :type x: array
    :param y: true labels
    :type y: array
    :param epsilon: small constant to add to log functions
    :type epsilon: constant
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The binary cross entropy loss array.
    """
    f = _get_framework(x, f=f)
    x = f.clip(x, epsilon, 1-epsilon)
    return -(y * f.log(x) + (1 - y) * f.log(1 - x))
