"""
Collection of linear algebra Ivy functions.
"""
# global
from typing import Union, Optional, Tuple, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


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
    return _cur_framework(x, f=f).svd(x)



def matrix_norm(x, p=2, axes=None, keepdims=False, f=None):
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
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Matrix norm of the array at specified axes.
    """
    return _cur_framework(x, f=f).matrix_norm(x, p, axes, keepdims)


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
    return _cur_framework(x, f=f).inv(x)


def pinv(x, f=None):
    """
    Computes the pseudo inverse of x matrix.

    :param x: Matrix to be pseudo inverted.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: pseudo inverse of the matrix x.
    """
    return _cur_framework(x, f=f).pinv(x)



def vector_to_skew_symmetric_matrix(vector):
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
    return _cur_framework(vector, f=f).vector_to_skew_symmetric_matrix(vector)


def cholesky(x):
    """
    Computes the cholesky decomposition of the x matrix.

    :param x: Matrix to be decomposed.
    :type x: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: cholesky decomposition of the matrix x.
    """
    return _cur_framework(x).cholesky(x)


def cross(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray], /, *,
          axis: Optional[int] = -1) -> ivy.Array:
    """
    Compute and return the cross product of 3-element vectors, it must have the same shape as b
    :param axis: the axis (dimension) of a and b containing the vector for which to compute the cross
    product default is -1
    :type  axis: int
    :param x1: first input, should have a numeric data type
    :type x1: array
    :param x2: second input, should have a numeric data type
    :type x2: array
    :return: an array that contains the cross products
    """
    return _cur_framework(x1).cross(x1, x2, axis)
