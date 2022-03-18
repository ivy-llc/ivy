"""
Collection of linear algebra Ivy functions.
"""

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def vector_to_skew_symmetric_matrix(vector):
    """
    Given vector :math:`\mathbf{a}\in\mathbb{R}^3`, return associated skew-symmetric matrix
    :math:`[\mathbf{a}]_×\in\mathbb{R}^{3×3}` satisfying :math:`\mathbf{a}×\mathbf{b}=[\mathbf{a}]_×\mathbf{b}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product>`_

    :param vector: Vector to convert *[batch_shape,3]*.
    :type vector: array
    :return: Skew-symmetric matrix *[batch_shape,3,3]*.
    """
    return _cur_framework(vector).vector_to_skew_symmetric_matrix(vector)
