# global
from typing import Union, Optional, Tuple, Literal, List
from collections import namedtuple

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework
inf = float('inf')


# Array API Standard #
# -------------------#


def eigh(x: Union[ivy.Array, ivy.NativeArray]) \
        -> ivy.Array:
    """
    Returns an eigendecomposition x = QLQᵀ of a symmetric matrix (or a stack of symmetric matrices) ``x``, where ``Q`` is an orthogonal matrix (or a stack of matrices) and ``L`` is a vector (or a stack of vectors).
    .. note::
       The function ``eig`` will be added in a future version of the specification, as it requires complex number support.
    ..
      NOTE: once complex numbers are supported, each square matrix must be Hermitian.
    .. note::
       Whether an array library explicitly checks whether an input array is a symmetric matrix (or a stack of symmetric matrices) is implementation-defined.
    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Must have a floating-point data type.
    Returns
    -------
    out: Tuple[array]
        a namedtuple (``eigenvalues``, ``eigenvectors``) whose
        -   first element must have the field name ``eigenvalues`` (corresponding to ``L`` above) and must be an array consisting of computed eigenvalues. The array containing the eigenvalues must have shape ``(..., M)``.
        -   second element have have the field name ``eigenvectors`` (corresponding to ``Q`` above) and must be an array where the columns of the inner most matrices contain the computed eigenvectors. These matrices must be orthogonal. The array containing the eigenvectors must have shape ``(..., M, M)``.
        Each returned array must have the same floating-point data type as ``x``.
    .. note::
       Eigenvalue sort order is left unspecified and is thus implementation-dependent.
    """
    return _cur_framework(x).eigh(x)   
         
         

def pinv(x: Union[ivy.Array, ivy.NativeArray],
         rtol: Optional[Union[float, Tuple[float]]] = None) \
        -> ivy.Array:
    """
    Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) ``x``.
    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Should have a floating-point data type.
    rtol: Optional[Union[float, array]]
        relative tolerance for small singular values. Singular values approximately less than or equal to ``rtol * largest_singular_value`` are set to zero. If a ``float``, the value is equivalent to a zero-dimensional array having a floating-point data type determined by :ref:`type-promotion` (as applied to ``x``) and must be broadcast against each matrix. If an ``array``, must have a floating-point data type and must be compatible with ``shape(x)[:-2]`` (see :ref:`broadcasting`). If ``None``, the default value is ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated with the floating-point data type determined by :ref:`type-promotion` (as applied to ``x``). Default: ``None``.
    Returns
    -------
    out: array
        an array containing the pseudo-inverses. The returned array must have a floating-point data type determined by :ref:`type-promotion` and must have shape ``(..., N, M)`` (i.e., must have the same shape as ``x``, except the innermost two dimensions must be transposed).
    """
    return _cur_framework(x).pinv(x, rtol)


def matrix_transpose(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Transposes a matrix (or a stack of matrices) ``x``.
    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    Returns
    -------
    out: array
        an array containing the transpose for each matrix and having shape ``(..., N, M)``. The returned array must have the same data type as ``x``.
    """
    return _cur_framework(x).matrix_transpose(x)


# noinspection PyShadowingBuiltins
def vector_norm(x: Union[ivy.Array, ivy.NativeArray],
                axis: Optional[Union[int, Tuple[int]]] = None,
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, -inf]] = 2)\
        -> ivy.Array:

    """
    Computes the vector norm of a vector (or batch of vectors) ``x``.

    Parameters
    ----------
    x:
        input array. Should have a floating-point data type.
    axis:
        If an integer, ``axis`` specifies the axis (dimension) along which to compute vector norms. If an n-tuple, ``axis`` specifies the axes (dimensions) along which to compute batched vector norms. If ``None``, the vector norm must be computed over all array values (i.e., equivalent to computing the vector norm of a flattened array). Negative indices must be supported. Default: ``None``.
    keepdims:
        If ``True``, the axes (dimensions) specified by ``axis`` must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the axes (dimensions) specified by ``axis`` must not be included in the result. Default: ``False``.
    ord:
        order of the norm. The following mathematical norms must be supported:
        +------------------+----------------------------+
        | ord              | description                |
        +==================+============================+
        | 1                | L1-norm (Manhattan)        |
        +------------------+----------------------------+
        | 2                | L2-norm (Euclidean)        |
        +------------------+----------------------------+
        | inf              | infinity norm              |
        +------------------+----------------------------+
        | (int,float >= 1) | p-norm                     |
        +------------------+----------------------------+
        The following non-mathematical "norms" must be supported:
        +------------------+--------------------------------+
        | ord              | description                    |
        +==================+================================+
        | 0                | sum(a != 0)                    |
        +------------------+--------------------------------+
        | -1               | 1./sum(1./abs(a))              |
        +------------------+--------------------------------+
        | -2               | 1./sqrt(sum(1./abs(a)\*\*2))   |
        +------------------+--------------------------------+
        | -inf             | min(abs(a))                    |
        +------------------+--------------------------------+
        | (int,float < 1)  | sum(abs(a)\*\*ord)\*\*(1./ord) |
        +------------------+--------------------------------+
        Default: ``2``.

    Returns
    -------
    out:
        an array containing the vector norms. If ``axis`` is ``None``, the returned array must be a zero-dimensional array containing a vector norm. If ``axis`` is a scalar value (``int`` or ``float``), the returned array must have a rank which is one less than the rank of ``x``. If ``axis`` is a ``n``-tuple, the returned array must have a rank which is ``n`` less than the rank of ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

    if ord == -float('inf'):
        return ivy.reduce_min(ivy.abs(x), axis, keepdims)
    elif ord == float('inf'):
        return ivy.reduce_max(ivy.abs(x), axis, keepdims)
    elif ord == 0:
        return ivy.sum(ivy.cast(x != 0, 'float32'), axis, keepdims)
    x_raised = x ** ord
    return ivy.sum(x_raised, axis, keepdims) ** (1/ord)


def svd(x:Union[ivy.Array,ivy.NativeArray],full_matrices: bool = True)->Union[ivy.Array, Tuple[ivy.Array,...]]:

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
    return _cur_framework(x).svd(x,full_matrices)


def outer(x1: Union[ivy.Array, ivy.NativeArray],
          x2: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    returns the outer product of two vectors x1 and x2.
    
    Parameters
    ----------
    x1 (array) – first one-dimensional input array of size N. Should have a numeric data type.
    a(M,) array_like
    First input vector. Input is flattened if not already 1-dimensional.

    x2 (array) – second one-dimensional input array of size M. Should have a numeric data type.
    b(N,) array_like
    Second input vector. Input is flattened if not already 1-dimensional.


    Returns
    -------
    out (array) – a two-dimensional array containing the outer product and whose shape is (N, M).
    The returned array must have a data type determined by Type Promotion Rules.
    out(M, N) ndarray, optional
    A location where the result is stored
    """
    return _cur_framework(x1, x2).outer(x1, x2)


def diagonal(x: ivy.Array,
             offset: int = 0,
             axis1: int = -2,
             axis2: int = -1) -> ivy.Array:
    """
    Returns the specified diagonals of a matrix (or a stack of matrices) ``x``.
    Parameters
    ----------
    x:
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    offset:
        offset specifying the off-diagonal relative to the main diagonal.
        - ``offset = 0``: the main diagonal.
        - ``offset > 0``: off-diagonal above the main diagonal.
        - ``offset < 0``: off-diagonal below the main diagonal.
        Default: `0`.
    axis1:
        axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken.
        Defaults to first axis (0).
    axis2:
        axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken.
        Defaults to second axis (1).

    Returns
    -------
    out:
        an array containing the diagonals and whose shape is determined by removing the last two dimensions and appending a dimension equal to the size of the resulting diagonals. The returned array must have the same data type as ``x``.
    """
    return _cur_framework(x).diagonal(x, offset, axis1=axis1, axis2=axis2)


def pinv(x):
    """
    Computes the pseudo inverse of x matrix.

    :param x: Matrix to be pseudo inverted.
    :type x: array
    :return: pseudo inverse of the matrix x.
    """
    return _cur_framework(x).pinv(x)


def cholesky(x):
    """
    Computes the cholesky decomposition of the x matrix.

    :param x: Matrix to be decomposed.
    :type x: array
    :return: cholesky decomposition of the matrix x.
    """
    return _cur_framework(x).cholesky(x)


def matrix_norm(x: Union[ivy.Array, ivy.NativeArray],
                ord: Optional[Union[int, float, Literal[inf, - inf, 'fro', 'nuc']]] = 'fro',
                keepdims: bool = False)\
        -> ivy.Array:
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
    return _cur_framework(x).matrix_norm(x, ord, keepdims)


def qr(x: ivy.Array,
       mode: str = 'reduced') -> namedtuple('qr', ['Q', 'R']):
    """
    Returns the qr decomposition x = QR of a full column rank matrix (or a stack of matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an upper-triangular matrix (or a stack of matrices).
    Parameters
    ----------
    x:
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices of rank N. Should have a floating-point data type.
    mode:
        decomposition mode. Should be one of the following modes:
        - 'reduced': compute only the leading K columns of q, such that q and r have dimensions (..., M, K) and (..., K, N), respectively, and where K = min(M, N).
        - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N), respectively.
        Default: 'reduced'.

    Returns
    -------
    out:
        a namedtuple (Q, R) whose
        - first element must have the field name Q and must be an array whose shape depends on the value of mode and contain matrices with orthonormal columns. If mode is 'complete', the array must have shape (..., M, M). If mode is 'reduced', the array must have shape (..., M, K), where K = min(M, N). The first x.ndim-2 dimensions must have the same size as those of the input array x.
        - second element must have the field name R and must be an array whose shape depends on the value of mode and contain upper-triangular matrices. If mode is 'complete', the array must have shape (..., M, N). If mode is 'reduced', the array must have shape (..., K, N), where K = min(M, N). The first x.ndim-2 dimensions must have the same size as those of the input x.
    """
    return _cur_framework(x).qr(x, mode)


def matmul(x1: Union[ivy.Array, ivy.NativeArray],
           x2: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Computes the matrix product.

    Parameters
    ----------
    x1:
        x1 (array) – first input array. Should have a numeric data type. Must have at least one dimension.

    x2:
        x2 (array) – second input array. Should have a numeric data type. Must have at least one dimension.

    Returns
    -------
    out(array):
        if both x1 and x2 are one-dimensional arrays having shape (N,), a zero-dimensional array containing the inner product as its only element.
        if x1 is a two-dimensional array having shape (M, K) and x2 is a two-dimensional array having shape (K, N), a two-dimensional array containing the conventional matrix product and having shape (M, N).
        if x1 is a one-dimensional array having shape (K,) and x2 is an array having shape (..., K, N), an array having shape (..., N) (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
        if x1 is an array having shape (..., M, K) and x2 is a one-dimensional array having shape (K,), an array having shape (..., M) (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
        if x1 is a two-dimensional array having shape (M, K) and x2 is an array having shape (..., K, N), an array having shape (..., M, N) and containing the conventional matrix product for each stacked matrix.
        if x1 is an array having shape (..., M, K) and x2 is a two-dimensional array having shape (K, N), an array having shape (..., M, N) and containing the conventional matrix product for each stacked matrix.
        if either x1 or x2 has more than two dimensions, an array having a shape determined by Broadcasting shape(x1)[:-2] against shape(x2)[:-2] and containing the conventional matrix product for each stacked matrix.

    Raises
    ------
        if either x1 or x2 is a zero-dimensional array.
        if x1 is a one-dimensional array having shape (K,), x2 is a one-dimensional array having shape (L,), and K != L.
        if x1 is a one-dimensional array having shape (K,), x2 is an array having shape (..., L, N), and K != L.
        if x1 is an array having shape (..., M, K), x2 is a one-dimensional array having shape (L,), and K != L.
        if x1 is an array having shape (..., M, K), x2 is an array having shape (..., L, N), and K != L.
    """
    return _cur_framework(x1).matmul(x1, x2)


def slodget(x: Union[ivy.Array, ivy.NativeArray],) \
            -> ivy.Array:
    """
    Computes the sign and natural logarithm of the determinant of an array.

    Parameters
    ----------
    x:
        This is a 2D array, and it has to be square

    Return
    ----------
    Out:

        This function returns two values -
            sign:
            A number representing the sign of the determinant.

            logdet:
            The natural log of the absolute value of the determinant.

    """
    return _cur_framework(x).slodget(x)

def tensordot(x1: Union[ivy.Array, ivy.NativeArray],
              x2: Union[ivy.Array, ivy.NativeArray],
              axes: Union[int, Tuple[List[int], List[int]]] = 2) \
    -> ivy.Array:

    """
    Returns a tensor contraction of x1 and x2 over specific axes.

    :param x1: First input array. Should have a numeric data type.
    :type x1: array
    :param x2: second input array. Must be compatible with x1 for all non-contracted axes.
               Should have a numeric data type.
    :type x2: array
    :param axes: The axes to contract over.
    :type axes: int or tuple of ints, or tuple of sequences of ints.
                Default is 2.
    :return: The tensor contraction of x1 and x2 over the specified axes.
    """

    return _cur_framework(x1, x2).tensordot(x1, x2, axes)


def svdvals(x: Union[ivy.Array, ivy.NativeArray],) \
            -> ivy.Array:
    """
    Returns the singular values of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x:
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    Return
    ----------
    Out:
        array with shape ``(..., K)`` that contains the vector(s) of singular values of length ``K``, where K = min(M, N).
        The values are sorted in descending order by magnitude.

    """
    return _cur_framework(x).svdvals(x)


def trace(x: Union[ivy.Array, ivy.NativeArray], offset: int = 0)\
        -> ivy.Array:
    """
    Returns the sum along the specified diagonals of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Should have a numeric data type.
    offset
        offset specifying the off-diagonal relative to the main diagonal.
        -   ``offset = 0``: the main diagonal.
        -   ``offset > 0``: off-diagonal above the main diagonal.
        -   ``offset < 0``: off-diagonal below the main diagonal.
        
        Default: ``0``.

     Returns
     -------
     ret
         an array containing the traces and whose shape is determined by removing the last two dimensions and storing the traces in the last array dimension. For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``, then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where
         ::
           out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])
         The returned array must have the same data type as ``x``.
     
     Examples
     --------
     >>> x = ivy.array([[1.0, 2.0],[3.0, 4.0]])
     >>> offset = 0
     >>> y = ivy.trace(x, offset)
     >>> print(y)
     5.0
     
    """
    return _cur_framework(x).trace(x, offset)


def vecdot(x1: Union[ivy.Array, ivy.NativeArray],
           x2: Union[ivy.Array, ivy.NativeArray],
           axis: int = -1)\
        -> ivy.Array:
    """
    Computes the (vector) dot product of two arrays.
    Parameters
    ----------
    x1: array
        first input array. Should have a numeric data type.
    x2: array
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.
    axis:int
        axis over which to compute the dot product. Must be an integer on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of the shape determined according to :ref:`broadcasting`. If specified as a negative integer, the function must determine the axis along which to compute the dot product by counting backward from the last dimension (where ``-1`` refers to the last dimension). By default, the function must compute the dot product over the last axis. Default: ``-1``.
    Returns
    -------
    out: array
        if ``x1`` and ``x2`` are both one-dimensional arrays, a zero-dimensional containing the dot product; otherwise, a non-zero-dimensional array containing the dot products and having rank ``N-1``, where ``N`` is the rank (number of dimensions) of the shape determined according to :ref:`broadcasting`. The returned array must have a data type determined by :ref:`type-promotion`.
    **Raises**
    -   if provided an invalid ``axis``.
    -   if the size of the axis over which to compute the dot product is not the same for both ``x1`` and ``x2``.
    """

    return _cur_framework(x1).vecdot(x1, x2, axis)


def det(x: ivy.Array) \
    -> ivy.Array:
    """
    Returns the determinant of a square matrix (or a stack of square matrices) x.

    :param x:  input array having shape (..., M, M) and whose innermost two dimensions form square matrices. Should
               have a floating-point data type.
    :return :  if x is a two-dimensional array, a zero-dimensional array containing the determinant; otherwise, a non-zero
               dimensional array containing the determinant for each square matrix. The returned array must have the same data type as x.
    """
    return _cur_framework(x).det(x)


def cholesky(x: Union[ivy.Array, ivy.NativeArray], 
             upper: bool = False) -> ivy.Array:
    """
    Computes the cholesky decomposition of the x matrix.

    :param x:  input array having shape (..., M, M) and whose innermost two dimensions form square symmetric
     positive-definite matrices. Should have a floating-point data type.
    :type x: array
    :param upper:  If True, the result must be the upper-triangular Cholesky factor U. If False, the result
     must be the lower-triangular Cholesky factor L. Default: False.
    :type upper: bool
    :return out: an array containing the Cholesky factors for each square matrix.
     If upper is False, the returned array must contain lower-triangular matrices; otherwise,
      the returned array must contain upper-triangular matrices. 
      The returned array must have a floating-point data type determined by Type Promotion Rules 
      and must have the same shape as x.
    :type out: array
    """
    return  _cur_framework(x).cholesky(x, upper)


def eigvalsh(x: Union[ivy.Array, ivy.NativeArray], /) -> ivy.Array:
    """
    Return the eigenvalues of a symmetric matrix (or a stack of symmetric matrices) x.
    :param x: input array having shape (..., M, M) and whose innermost two dimensions form square matrices.
              Must have floating-point data type.

    :return: an array containing the computed eigenvalues. The returned array must have shape (..., M) and
             have the same data type as x.
    """
    return _cur_framework(x).eigvalsh(x)


def inv(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Returns the multiplicative inverse of a square matrix (or a stack of square matrices) x.

    Parameters
    x (array) : input array having shape (..., M, M) and whose innermost two dimensions form square matrices.
    Should have a floating-point data type.

    Returns
    out (array) : an array containing the multiplicative inverses.
    The returned array must have a floating-point data type determined by Type Promotion Rules and must have the same shape as x.
    """
    return _cur_framework(x).inv(x)


def matrix_rank(vector: Union[ivy.Array, ivy.NativeArray],
                rtol: Optional[Union[float, Tuple[float]]] = None) \
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).

    Parameters:
    x:
    (array) – input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices. Should have a floating-point data type.

    rtol:
    (Optional[Union[float, array]]) – relative tolerance for small singular values.
    Singular values approximately less than or equal to rtol * largest_singular_value are set to zero.
    If a float, the value is equivalent to a zero-dimensional array having a floating-point data type determined by Type Promotion Rules (as applied to x) and must be broadcast against each matrix.
    If an array, must have a floating-point data type and must be compatible with shape(x)[:-2] .
    If None, the default value is max(M, N) * eps, where eps must be the machine epsilon associated with the floating-point data type determined by Type Promotion Rules (as applied to x). Default: None.

    Returns:
    out:
    (array) – an array containing the ranks.

    """
    return _cur_framework(vector).matrix_rank(vector, rtol)


def cross(x1: Union[ivy.Array, ivy.NativeArray],
          x2: Union[ivy.Array, ivy.NativeArray],
          axis: int = - 1) -> ivy.Array:
    """
    The cross product of 3-element vectors. If x1 and x2 are multi-dimensional arrays 
    (i.e., both have a rank greater than 1), then the cross-product of each pair of corresponding 
    3-element vectors is independently computed.

    Parameters
    :param x1: first input array. Should have a numeric data type.
    :type x1: array
    :param x2: second input array. Must have the same shape as x1. Should have a numeric data type.
    :type x2: array
    :param axis: the axis (dimension) of x1 and x2 containing the vectors for which to compute the cross product.
     If set to -1, the function computes the cross product for vectors defined by the last axis (dimension).
      Default: -1.
    :type axis: int
    :returnout: an array containing the cross products. The returned array must have a data type determined
     by Type Promotion Rules.
    :type out: array
    """
    return _cur_framework(x1).cross(x1,x2,axis)


# Extra #
# ------#

def vector_to_skew_symmetric_matrix(vector: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Given vector :math:`\mathbf{a}\in\mathbb{R}^3`, return associated skew-symmetric matrix
    :math:`[\mathbf{a}]_×\in\mathbb{R}^{3×3}` satisfying :math:`\mathbf{a}×\mathbf{b}=[\mathbf{a}]_×\mathbf{b}`.\n
    `[reference] <https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product>`_

    :param vector: Vector to convert *[batch_shape,3]*.
    :type vector: array
    :return: Skew-symmetric matrix *[batch_shape,3,3]*.
    """
    return _cur_framework(vector).vector_to_skew_symmetric_matrix(vector)
