"""
Collection of linear algebra Ivy functions.
"""

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def svd(x):
    """
    Singular Value Decomposition.
    When x is a 2D array, it is factorized as u @ numpy.diag(s) @ vh = (u * s) @ vh, where u and vh are 2D unitary
    arrays and s is a 1D array of a’s singular values. When x is higher-dimensional, SVD is applied in batched mode.

    :param x: Input array with number of dimensions >= 2.
    :type x: array
    :return:
        u -> { (…, M, M), (…, M, K) } array \n
        Unitary array(s). The first (number of dims - 2) dimensions have the same size as those of the input a.
        The size of the last two dimensions depends on the value of full_matrices.

        s -> (…, K) array \n
        Vector(s) with the singular values, within each vector sorted in descending ord.
        The first (number of dims - 2) dimensions have the same size as those of the input a.

        vh -> { (…, N, N), (…, K, N) } array \n
        Unitary array(s). The first (number of dims - 2) dimensions have the same size as those of the input a.
        The size of the last two dimensions depends on the value of full_matrices.
    """
    return _cur_framework(x).svd(x)




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


def inv(x):
    """
    Computes the (multiplicative) inverse of x matrix.
    Given a square matrix x, returns the matrix x_inv satisfying dot(x, x_inv) = dot(x_inv, x) = eye(x.shape[0]).

    :param x: Matrix to be inverted.
    :type x: array
    :return: (Multiplicative) inverse of the matrix x.
    """
    return _cur_framework(x).inv(x)


def pinv(x):
    """
    Computes the pseudo inverse of x matrix.

    :param x: Matrix to be pseudo inverted.
    :type x: array
    :return: pseudo inverse of the matrix x.
    """
    return _cur_framework(x).pinv(x)


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
