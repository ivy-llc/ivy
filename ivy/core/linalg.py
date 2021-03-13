"""
Collection of linear algebra Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def svd(x, f=None):
    """
    Singular Value Decomposition.
    When x is a 2D array, it is factorized as u @ numpy.diag(s) @ vh = (u * s) @ vh, where u and vh are 2D unitary
    arrays and s is a 1D array of a’s singular values. When x is higher-dimensional, SVD is applied in batched mode.

    :param x: Input array with number of dimensions >= 2.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
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
    return _get_framework(x, f=f).svd(x)


# noinspection PyShadowingBuiltins
def norm(x, ord=2, axis=-1, keepdims=False, f=None):
    """
    Matrix or vector norm.
    This function is able to return ord-1 and ord-2 vector-norms and matrix-norms.

    :param x: Input array. If axis is None, x must be 1-D or 2-D.
    :type x: array
    :param ord: Order of the norm.
    :type ord: int
    :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a
                    2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are
                    computed.
    :type axis: {int, 2-sequence of ints}, optional
    :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with
                        size one. With this option the result will broadcast correctly against the original x.
                        Default is False.
    :type keepdims: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Norm of the matrix or vector(s).
    """
    return _get_framework(x, f=f).norm(x, ord, axis, keepdims)


def inv(x, f=None):
    """
    Computes the (multiplicative) inverse of x matrix.
    Given a square matrix x, returns the matrix x_inv satisfying dot(x, x_inv) = dot(x_inv, x) = eye(x.shape[0]).

    :param x: Matrix to be inverted.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: (Multiplicative) inverse of the matrix x.
    """
    return _get_framework(x, f=f).inv(x)


def pinv(x, f=None):
    """
    Computes the pseudo inverse of x matrix.

    :param x: Matrix to be pseudo inverted.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: pseudo inverse of the matrix x.
    """
    return _get_framework(x, f=f).pinv(x)


def vector_to_skew_symmetric_matrix(vector, f=None):
    """
    Given vector :math:`\mathbf{a}\in\mathbb{R}^3`, return associated skew-symmetric matrix
    :math:`[\mathbf{a}]_×\in\mathbb{R}^{3×3}` satisfying :math:`\mathbf{a}×\mathbf{b}=[\mathbf{a}]_×\mathbf{b}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product>`_

    :param vector: Vector to convert *[batch_shape,3]*.
    :type vector: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Skew-symmetric matrix *[batch_shape,3,3]*.
    """
    return _get_framework(vector, f=f).vector_to_skew_symmetric_matrix(vector)
