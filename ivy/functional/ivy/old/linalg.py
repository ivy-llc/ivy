"""
Collection of linear algebra Ivy functions.
"""

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework




def matrix_norm(x, p=2, axes=None, keepdims=False):
    """
    Compute the matrix p-norm.

    :param x: Input array.
    :type x: array
    :param p: Order of the norm. Default is 2.
    :type p: int or str, optional
    :param axes: The axes of x along which to compute the matrix norms.
                 Default is None, in which case the last two dimensions are used.
    :type axes: sequence of ints, optional
    :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with
                     size one. With this option the result will broadcast correctly against the original x.
                     Default is False.
    :type keepdims: bool, optional
    :return: Matrix norm of the array at specified axes.
    """
    return _cur_framework(x).matrix_norm(x, p, axes, keepdims)


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

def cholesky(x):
    """
    Computes the cholesky decomposition of the x matrix.

    :param x: Matrix to be decomposed.
    :type x: array
    :return: cholesky decomposition of the matrix x.
    """
    return _cur_framework(x).cholesky(x)

def qr(x, mode="reduced", f=None):
    """
    Computes the qr decomposition of the x matrix.

    :param x: Matrix to be decomposed.
    :type x: array
    :param mode: The option to choose between the full and reduced QR decomposition.
    :type mode: str, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: qr decomposition of the matrix x.

    If x has shape (*, M, N) and considering K = min(M, N).
    mode = 'reduced' (default): Returns (Q, R) of shapes (*, M, K), (*, K, N) respectively.
    mode = 'complete': Returns (Q, R) of shapes (*, M, M), (*, M, N) respectively.
    mode = 'r': Computes only the reduced R. Returns (Q, R) with Q empty and R of shape (*, K, N).

    Please note that different frameworks support different modes. For example: mode='raw'
    is supported in numpy but not in pytorch. Similarly, tensorflow doesn't support 'r' mode.
    The 'reduced' and 'complete' modes are sufficient for most of the use-cases and are supported
    by all the frameworks.
    """
    return _cur_framework(x).qr(x, mode=mode)
