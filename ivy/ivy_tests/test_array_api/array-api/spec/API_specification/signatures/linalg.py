from ._types import Literal, Optional, Tuple, Union, Sequence, array
from .constants import inf

def cholesky(x: array, /, *, upper: bool = False) -> array:
    """
    Returns the lower (upper) Cholesky decomposition x = LLᵀ (x = UᵀU) of a symmetric positive-definite matrix (or a stack of matrices) ``x``, where ``L`` is a lower-triangular matrix or a stack of matrices (``U`` is an upper-triangular matrix or a stack of matrices).

    ..
      NOTE: once complex numbers are supported, each square matrix must be Hermitian.

    .. note::
       Whether an array library explicitly checks whether an input array is a symmetric positive-definite matrix (or a stack of matrices) is implementation-defined.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square symmetric positive-definite matrices. Should have a floating-point data type.
    upper: bool
        If ``True``, the result must be the upper-triangular Cholesky factor ``U``. If ``False``, the result must be the lower-triangular Cholesky factor ``L``. Default: ``False``.

    Returns
    -------
    out: array
        an array containing the Cholesky factors for each square matrix. If ``upper`` is ``False``, the returned array must contain lower-triangular matrices; otherwise, the returned array must contain upper-triangular matrices. The returned array must have a floating-point data type determined by :ref:`type-promotion` and must have the same shape as ``x``.
    """

def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    """
    Returns the cross product of 3-element vectors. If ``x1`` and ``x2`` are multi-dimensional arrays (i.e., both have a rank greater than ``1``), then the cross-product of each pair of corresponding 3-element vectors is independently computed.

    Parameters
    ----------
    x1: array
        first input array. Should have a numeric data type.
    x2: array
        second input array. Must have the same shape as ``x1``.  Should have a numeric data type.
    axis: int
        the axis (dimension) of ``x1`` and ``x2`` containing the vectors for which to compute the cross product. If set to ``-1``, the function computes the cross product for vectors defined by the last axis (dimension). Default: ``-1``.

    Returns
    -------
    out: array
        an array containing the cross products. The returned array must have a data type determined by :ref:`type-promotion`.
    """

def det(x: array, /) -> array:
    """
    Returns the determinant of a square matrix (or a stack of square matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Should have a floating-point data type.

    Returns
    -------
    out: array
        if ``x`` is a two-dimensional array, a zero-dimensional array containing the determinant; otherwise, a non-zero dimensional array containing the determinant for each square matrix. The returned array must have the same data type as ``x``.
    """

def diagonal(x: array, /, *, offset: int = 0) -> array:
    """
    Returns the specified diagonals of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    offset: int
        offset specifying the off-diagonal relative to the main diagonal.

        - ``offset = 0``: the main diagonal.
        - ``offset > 0``: off-diagonal above the main diagonal.
        - ``offset < 0``: off-diagonal below the main diagonal.

        Default: `0`.

    Returns
    -------
    out: array
        an array containing the diagonals and whose shape is determined by removing the last two dimensions and appending a dimension equal to the size of the resulting diagonals. The returned array must have the same data type as ``x``.
    """

def eigh(x: array, /) -> Tuple[array]:
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

def eigvalsh(x: array, /) -> array:
    """
    Returns the eigenvalues of a symmetric matrix (or a stack of symmetric matrices) ``x``.

    .. note::
       The function ``eigvals`` will be added in a future version of the specification, as it requires complex number support.

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
    out: array
        an array containing the computed eigenvalues. The returned array must have shape ``(..., M)`` and have the same data type as ``x``.

    .. note::
       Eigenvalue sort order is left unspecified and is thus implementation-dependent.
    """

def inv(x: array, /) -> array:
    """
    Returns the multiplicative inverse of a square matrix (or a stack of square matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Should have a floating-point data type.

    Returns
    -------
    out: array
        an array containing the multiplicative inverses. The returned array must have a floating-point data type determined by :ref:`type-promotion` and must have the same shape as ``x``.
    """

def matmul(x1: array, x2: array, /) -> array:
    """
    Alias for :func:`~signatures.linear_algebra_functions.matmul`.
    """

def matrix_norm(x: array, /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal[inf, -inf, 'fro', 'nuc']]] = 'fro') -> array:
    """
    Computes the matrix norm of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Should have a floating-point data type.
    keepdims: bool
        If ``True``, the last two axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the last two axes (dimensions) must not be included in the result. Default: ``False``.
    ord: Optional[Union[int, float, Literal[inf, -inf, 'fro', 'nuc']]]
        order of the norm. The following mathematical norms must be supported:

        +------------------+---------------------------------+
        | ord              | description                     |
        +==================+=================================+
        | 'fro'            | Frobenius norm                  |
        +------------------+---------------------------------+
        | 'nuc'            | nuclear norm                    |
        +------------------+---------------------------------+
        | 1                | max(sum(abs(x), axis=0))        |
        +------------------+---------------------------------+
        | 2                | largest singular value          |
        +------------------+---------------------------------+
        | inf              | max(sum(abs(x), axis=1))        |
        +------------------+---------------------------------+

        The following non-mathematical "norms" must be supported:

        +------------------+---------------------------------+
        | ord              | description                     |
        +==================+=================================+
        | -1               | min(sum(abs(x), axis=0))        |
        +------------------+---------------------------------+
        | -2               | smallest singular value         |
        +------------------+---------------------------------+
        | -inf             | min(sum(abs(x), axis=1))        |
        +------------------+---------------------------------+

        If ``ord=1``, the norm corresponds to the induced matrix norm where ``p=1`` (i.e., the maximum absolute value column sum).

        If ``ord=2``, the norm corresponds to the induced matrix norm where ``p=inf`` (i.e., the maximum absolute value row sum).

        If ``ord=inf``, the norm corresponds to the induced matrix norm where ``p=2`` (i.e., the largest singular value).

        Default: ``'fro'``.

    Returns
    -------
    out: array
        an array containing the norms for each ``MxN`` matrix. If ``keepdims`` is ``False``, the returned array must have a rank which is two less than the rank of ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

def matrix_power(x: array, n: int, /) -> array:
    """
    Raises a square matrix (or a stack of square matrices) ``x`` to an integer power ``n``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Should have a floating-point data type.
    n: int
        integer exponent.

    Returns
    -------
    out: array
        if ``n`` is equal to zero, an array containing the identity matrix for each square matrix. If ``n`` is less than zero, an array containing the inverse of each square matrix raised to the absolute value of ``n``, provided that each square matrix is invertible. If ``n`` is greater than zero, an array containing the result of raising each square matrix to the power ``n``. The returned array must have the same shape as ``x`` and a floating-point data type determined by :ref:`type-promotion`.
    """

def matrix_rank(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    """
    Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Should have a floating-point data type.
    rtol: Optional[Union[float, array]]
        relative tolerance for small singular values. Singular values approximately less than or equal to ``rtol * largest_singular_value`` are set to zero. If a ``float``, the value is equivalent to a zero-dimensional array having a floating-point data type determined by :ref:`type-promotion` (as applied to ``x``) and must be broadcast against each matrix. If an ``array``, must have a floating-point data type and must be compatible with ``shape(x)[:-2]`` (see :ref:`broadcasting`). If ``None``, the default value is ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated with the floating-point data type determined by :ref:`type-promotion` (as applied to ``x``). Default: ``None``.

    Returns
    -------
    out: array
        an array containing the ranks. The returned array must have a floating-point data type determined by :ref:`type-promotion` and must have shape ``(...)`` (i.e., must have a shape equal to ``shape(x)[:-2]``).
    """

def matrix_transpose(x: array, /) -> array:
    """
    Alias for :func:`~signatures.linear_algebra_functions.matrix_transpose`.
    """

def outer(x1: array, x2: array, /) -> array:
    """
    Returns the outer product of two vectors ``x1`` and ``x2``.

    Parameters
    ----------
    x1: array
        first one-dimensional input array of size ``N``. Should have a numeric data type.
    x2: array
        second one-dimensional input array of size ``M``. Should have a numeric data type.

    Returns
    -------
    out: array
        a two-dimensional array containing the outer product and whose shape is ``(N, M)``. The returned array must have a data type determined by :ref:`type-promotion`.
    """

def pinv(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
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

def qr(x: array, /, *, mode: Literal['reduced', 'complete'] = 'reduced') -> Tuple[array, array]:
    """
    Returns the qr decomposition x = QR of a full column rank matrix (or a stack of matrices), where ``Q`` is an orthonormal matrix (or a stack of matrices) and ``R`` is an upper-triangular matrix (or a stack of matrices).

    .. note::
       Whether an array library explicitly checks whether an input array is a full column rank matrix (or a stack of full column rank matrices) is implementation-defined.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices of rank ``N``. Should have a floating-point data type.
    mode: Literal['reduced', 'complete']
        decomposition mode. Should be one of the following modes:

        -   ``'reduced'``: compute only the leading ``K`` columns of ``q``, such that ``q`` and ``r`` have dimensions ``(..., M, K)`` and ``(..., K, N)``, respectively, and where ``K = min(M, N)``.
        -   ``'complete'``: compute ``q`` and ``r`` with dimensions ``(..., M, M)`` and ``(..., M, N)``, respectively.

        Default: ``'reduced'``.

    Returns
    -------
    out: Tuple[array, array]
        a namedtuple ``(Q, R)`` whose

        -   first element must have the field name ``Q`` and must be an array whose shape depends on the value of ``mode`` and contain matrices with orthonormal columns. If ``mode`` is ``'complete'``, the array must have shape ``(..., M, M)``. If ``mode`` is ``'reduced'``, the array must have shape ``(..., M, K)``, where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions must have the same size as those of the input array ``x``.
        -   second element must have the field name ``R`` and must be an array whose shape depends on the value of ``mode`` and contain upper-triangular matrices. If ``mode`` is ``'complete'``, the array must have shape ``(..., M, N)``. If ``mode`` is ``'reduced'``, the array must have shape ``(..., K, N)``, where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions must have the same size as those of the input ``x``.

        Each returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

def slogdet(x: array, /) -> Tuple[array, array]:
    """
    Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) ``x``.

    .. note::
       The purpose of this function is to calculate the determinant more accurately when the determinant is either very small or very large, as calling ``det`` may overflow or underflow.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Should have a floating-point data type.

    Returns
    -------
    out: Tuple[array, array]
        a namedtuple (``sign``, ``logabsdet``) whose

        -   first element must have the field name ``sign`` and must be an array containing a number representing the sign of the determinant for each square matrix.
        -   second element must have the field name ``logabsdet`` and must be an array containing the determinant for each square matrix.

        For a real matrix, the sign of the determinant must be either ``1``, ``0``, or ``-1``.

        Each returned array must have shape ``shape(x)[:-2]`` and a floating-point data type determined by :ref:`type-promotion`.

        .. note::
           If a determinant is zero, then the corresponding ``sign`` should be ``0`` and ``logabsdet`` should be ``-infinity``; however, depending on the underlying algorithm, the returned result may differ. In all cases, the determinant should be equal to ``sign * exp(logsabsdet)`` (although, again, the result may be subject to numerical precision errors).
    """

def solve(x1: array, x2: array, /) -> array:
    """
    Returns the solution to the system of linear equations represented by the well-determined (i.e., full rank) linear matrix equation ``AX = B``.

    .. note::
       Whether an array library explicitly checks whether an input array is full rank is implementation-defined.

    Parameters
    ----------
    x1: array
        coefficient array ``A`` having shape ``(..., M, M)`` and whose innermost two dimensions form square matrices. Must be of full rank (i.e., all rows or, equivalently, columns must be linearly independent). Should have a floating-point data type.
    x2: array
        ordinate (or "dependent variable") array ``B``. If ``x2`` has shape ``(M,)``, ``x2`` is equivalent to an array having shape ``(..., M, 1)``. If ``x2`` has shape ``(..., M, K)``, each column ``k`` defines a set of ordinate values for which to compute a solution, and ``shape(x2)[:-1]`` must be compatible with ``shape(x1)[:-1]`` (see :ref:`broadcasting`). Should have a floating-point data type.

    Returns
    -------
    out: array
        an array containing the solution to the system ``AX = B`` for each square matrix. The returned array must have the same shape as ``x2`` (i.e., the array corresponding to ``B``) and must have a floating-point data type determined by :ref:`type-promotion`.
    """

def svd(x: array, /, *, full_matrices: bool = True) -> Union[array, Tuple[array, ...]]:
    """
    Returns a singular value decomposition A = USVh of a matrix (or a stack of matrices) ``x``, where ``U`` is a matrix (or a stack of matrices) with orthonormal columns, ``S`` is a vector of non-negative numbers (or stack of vectors), and ``Vh`` is a matrix (or a stack of matrices) with orthonormal rows.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form matrices on which to perform singular value decomposition. Should have a floating-point data type.
    full_matrices: bool
        If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has shape ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``, compute on the leading ``K`` singular vectors, such that ``U`` has shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where ``K = min(M, N)``. Default: ``True``.

    Returns
    -------
    ..
      NOTE: once complex numbers are supported, each square matrix must be Hermitian.
    out: Union[array, Tuple[array, ...]]
        a namedtuple ``(U, S, Vh)`` whose

        -   first element must have the field name ``U`` and must be an array whose shape depends on the value of ``full_matrices`` and contain matrices with orthonormal columns (i.e., the columns are left singular vectors). If ``full_matrices`` is ``True``, the array must have shape ``(..., M, M)``. If ``full_matrices`` is ``False``, the array must have shape ``(..., M, K)``, where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions must have the same shape as those of the input ``x``.
        -   second element must have the field name ``S`` and must be an array with shape ``(..., K)`` that contains the vector(s) of singular values of length ``K``, where ``K = min(M, N)``. For each vector, the singular values must be sorted in descending order by magnitude, such that ``s[..., 0]`` is the largest value, ``s[..., 1]`` is the second largest value, et cetera. The first ``x.ndim-2`` dimensions must have the same shape as those of the input ``x``.
        -   third element must have the field name ``Vh`` and must be an array whose shape depends on the value of ``full_matrices`` and contain orthonormal rows (i.e., the rows are the right singular vectors and the array is the adjoint). If ``full_matrices`` is ``True``, the array must have shape ``(..., N, N)``. If ``full_matrices`` is ``False``, the array must have shape ``(..., K, N)`` where ``K = min(M, N)``. The first ``x.ndim-2`` dimensions must have the same shape as those of the input ``x``.

        Each returned array must have the same floating-point data type as ``x``.
    """

def svdvals(x: array, /) -> array:
    """
    Returns the singular values of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form matrices on which to perform singular value decomposition. Should have a floating-point data type.

    Returns
    -------
    out: array
        an array with shape ``(..., K)`` that contains the vector(s) of singular values of length ``K``, where ``K = min(M, N)``. For each vector, the singular values must be sorted in descending order by magnitude, such that ``s[..., 0]`` is the largest value, ``s[..., 1]`` is the second largest value, et cetera. The first ``x.ndim-2`` dimensions must have the same shape as those of the input ``x``. The returned array must have the same floating-point data type as ``x``.
    """

def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> array:
    """
    Alias for :func:`~signatures.linear_algebra_functions.tensordot`.
    """

def trace(x: array, /, *, offset: int = 0) -> array:
    """
    Returns the sum along the specified diagonals of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Should have a numeric data type.
    offset: int
        offset specifying the off-diagonal relative to the main diagonal.

        -   ``offset = 0``: the main diagonal.
        -   ``offset > 0``: off-diagonal above the main diagonal.
        -   ``offset < 0``: off-diagonal below the main diagonal.

        Default: ``0``.

    Returns
    -------
    out: array
        an array containing the traces and whose shape is determined by removing the last two dimensions and storing the traces in the last array dimension. For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``, then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

        ::

          out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

        The returned array must have the same data type as ``x``.
    """

def vecdot(x1: array, x2: array, /, *, axis: int = None) -> array:
    """
    Alias for :func:`~signatures.linear_algebra_functions.vecdot`.
    """

def vector_norm(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Union[int, float, Literal[inf, -inf]] = 2) -> array:
    """
    Computes the vector norm of a vector (or batch of vectors) ``x``.

    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    axis: Optional[Union[int, Tuple[int, ...]]]
        If an integer, ``axis`` specifies the axis (dimension) along which to compute vector norms. If an n-tuple, ``axis`` specifies the axes (dimensions) along which to compute batched vector norms. If ``None``, the vector norm must be computed over all array values (i.e., equivalent to computing the vector norm of a flattened array). Negative indices must be supported. Default: ``None``.
    keepdims: bool
        If ``True``, the axes (dimensions) specified by ``axis`` must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the axes (dimensions) specified by ``axis`` must not be included in the result. Default: ``False``.
    ord: Union[int, float, Literal[inf, -inf]]
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
    out: array
        an array containing the vector norms. If ``axis`` is ``None``, the returned array must be a zero-dimensional array containing a vector norm. If ``axis`` is a scalar value (``int`` or ``float``), the returned array must have a rank which is one less than the rank of ``x``. If ``axis`` is a ``n``-tuple, the returned array must have a rank which is ``n`` less than the rank of ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.
    """

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']
