"""
Collection of Ivy loss functions.
"""

# local
import ivy


# noinspection PyUnresolvedReferences
def binary_cross_entropy(x, y, epsilon=1e-7):
    """
    Computes the cross-entropy loss between true labels and predicted labels.

    :param x: Predicted labels
    :type x: array
    :param y: true labels
    :type y: array
    :param epsilon: small constant to add to log functions
    :type epsilon: constant
    :return: The binary cross entropy loss array.
    """
    x = ivy.clip(x, epsilon, 1-epsilon)
    return -(y * ivy.log(x) + (1 - y) * ivy.log(1 - x))
