# global
import logging
from typing import Union, Optional, Tuple, List, Sequence, Literal

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_array_function,
    handle_device,
    inputs_to_ivy_arrays,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions

# Helpers #
# ------- #


def _check_valid_dimension_size(std):
    ivy.utils.assertions.check_dimensions(std)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_array_function
def eigh_tridiagonal(
    alpha: Union[ivy.Array, ivy.NativeArray],
    beta: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    eigvals_only: bool = True,
    select: str = "a",
    select_range: Optional[
        Union[Tuple[int, int], List[int], ivy.Array, ivy.NativeArray]
    ] = None,
    tol: Optional[float] = None,
) -> Union[ivy.Array, Tuple[ivy.Array, ivy.Array]]:
    """Compute the eigenvalues and eigenvectors of a Hermitian tridiagonal
    matrix.

    Parameters
    ----------
    alpha
        A real or complex array of shape (n), the diagonal elements of the
        matrix. If alpha is complex, the imaginary part is ignored
        (assumed zero) to satisfy the requirement that the matrix be Hermitian.
    beta
        A real or complex array of shape (n-1), containing the elements of
        the first super-diagonal of the matrix. If beta is complex, the first
        sub-diagonal of the matrix is assumed to be the conjugate of beta to
        satisfy the requirement that the matrix be Hermitian.
    eigvals_only
        If False, both eigenvalues and corresponding eigenvectors are
        computed. If True, only eigenvalues are computed. Default is True.
    select
        Optional string with values in {'a', 'v', 'i'} (default is 'a') that
        determines which eigenvalues to calculate: 'a': all eigenvalues.
        'v': eigenvalues in the interval (min, max] given by select_range.
        'i': eigenvalues with indices min <= i <= max.
    select_range
        Size 2 tuple or list or array specifying the range of eigenvalues to
        compute together with select. If select is 'a', select_range is ignored.
    tol
        Optional scalar. Ignored when backend is not Tensorflow. The absolute
        tolerance to which each eigenvalue is required. An eigenvalue
        (or cluster) is considered to have converged if it lies in an interval
        of this width. If tol is None (default), the value eps*|T|_2 is used
        where eps is the machine precision, and |T|_2 is the 2-norm of the matrix T.

    Returns
    -------
    eig_vals
        The eigenvalues of the matrix in non-decreasing order.
    eig_vectors
        If eigvals_only is False the eigenvectors are returned in the second output
        argument.

    Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> alpha = ivy.array([0., 1., 2.])
    >>> beta = ivy.array([0., 1.])
    >>> y = ivy.eigh_tridiagonal(alpha, beta)
    >>> print(y)
    ivy.array([0., 0.38196602, 2.61803389])

    >>> alpha = ivy.array([0., 1., 2.])
    >>> beta = ivy.array([0., 1.])
    >>> y = ivy.eigh_tridiagonal(alpha,
    ...     beta, select='v',
    ...     select_range=[0.2,3.0])
    >>> print(y)
    ivy.array([0.38196602, 2.61803389])

    >>> alpha = ivy.array([0., 1., 2., 3.])
    >>> beta = ivy.array([2., 1., 2.])
    >>> y = ivy.eigh_tridiagonal(alpha,
    ...     beta,
    ...     eigvals_only=False,
    ...     select='i',
    ...     select_range=[1,2],
    ...     tol=1.)
    >>> print(y)
    (ivy.array([0.38196602, 2.61803389]), ivy.array([[ 0.35048741, -0.56710052],
           [ 0.06693714, -0.74234426],
           [-0.74234426, -0.06693714],
           [ 0.56710052,  0.35048741]]))

    With :class:`ivy.Container` input:

    >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
    >>> beta = ivy.array([0.,2.])
    >>> y = ivy.eigh_tridiagonal(alpha, beta)
    >>> print(y)
    {
        a: ivy.array([-0.56155282, 0., 3.56155276]),
        b: ivy.array([0., 2., 4.])
    }

    >>> alpha = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([2., 2., 2.]))
    >>> beta = ivy.Container(a=ivy.array([0.,2.]), b=ivy.array([2.,2.]))
    >>> y = ivy.eigh_tridiagonal(alpha, beta)
    >>> print(y)
    {
        a: ivy.array([-0.56155282, 0., 3.56155276]),
        b: ivy.array([-0.82842714, 2., 4.82842731])
    }
    """
    x = ivy.diag(alpha)
    y = ivy.diag(beta, k=1)
    z = ivy.diag(beta, k=-1)
    w = x + y + z

    eigh_out = ivy.linalg.eigh(w)
    eigenvalues = eigh_out.eigenvalues
    eigenvectors = eigh_out.eigenvectors

    if select == "i":
        eigenvalues = eigenvalues[select_range[0] : select_range[1] + 1]
        eigenvectors = eigenvectors[:, select_range[0] : select_range[1] + 1]
    elif select == "v":
        condition = ivy.logical_and(
            eigenvalues.greater(select_range[0]),
            eigenvalues.less_equal(select_range[1]),
        )
        eigenvalues = eigenvalues[condition]
        eigenvectors = eigenvectors[:, condition]

    if eigvals_only:
        return eigenvalues
    return eigenvalues, eigenvectors


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def diagflat(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    offset: int = 0,
    padding_value: float = 0,
    align: str = "RIGHT_LEFT",
    num_rows: int = -1,
    num_cols: int = -1,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Return a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    x
        Input data, which is flattened and set as the k-th diagonal of the output.
    k
        Diagonal to set.
        Positive value means superdiagonal,
        0 refers to the main diagonal,
        and negative value means subdiagonal.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The 2-D output array.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([[1,2], [3,4]])
    >>> ivy.diagflat(x)
    ivy.array([[1, 0, 0, 0],
               [0, 2, 0, 0],
               [0, 0, 3, 0],
               [0, 0, 0, 4]])

    >>> x = ivy.array([1,2])
    >>> ivy.diagflat(x, k=1)
    ivy.array([[0, 1, 0],
               [0, 0, 2],
               [0, 0, 0]])
    """
    return current_backend(x).diagflat(
        x,
        offset=offset,
        padding_value=padding_value,
        align=align,
        num_rows=num_rows,
        num_cols=num_cols,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def kron(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Parameters
    ----------
    a
        First input array.
    b
        Second input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Array representing the Kronecker product of the input arrays.

    Examples
    --------
    >>> a = ivy.array([1,2])
    >>> b = ivy.array([3,4])
    >>> ivy.kron(a, b)
    ivy.array([3, 4, 6, 8])
    """
    return current_backend(a, b).kron(a, b, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def matrix_exp(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the matrix exponential of a square matrix.

    Parameters
    ----------
    a
        Square matrix.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        the matrix exponential of the input.

    Examples
    --------
        >>> x = ivy.array([[[1., 0.],
                            [0., 1.]],
                            [[2., 0.],
                            [0., 2.]]])
        >>> ivy.matrix_exp(x)
        ivy.array([[[2.7183, 1.0000],
                    [1.0000, 2.7183]],
                    [[7.3891, 1.0000],
                    [1.0000, 7.3891]]])
    """
    return current_backend(x).matrix_exp(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_device
def eig(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple[ivy.Array]:
    """Compute eigenvalies and eigenvectors of x. Returns a tuple with two
    elements: first is the set of eigenvalues, second is the set of
    eigenvectors.

    Parameters
    ----------
    x
        An array of shape (..., N, N).

    Returns
    -------
    w
        Not necessarily ordered array(..., N) of eigenvalues in complex type.
    v
        An array(..., N, N) of normalized (unit “length”) eigenvectors,
        the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_.
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` inputs:
    >>> x = ivy.array([[1,2], [3,4]])
    >>> w, v = ivy.eig(x)
    >>> w; v
    ivy.array([-0.37228132+0.j,  5.37228132+0.j])
    ivy.array([[-0.82456484+0.j, -0.41597356+0.j],
               [ 0.56576746+0.j, -0.90937671+0.j]])

    >>> x = ivy.array([[[1,2], [3,4]], [[5,6], [5,6]]])
    >>> w, v = ivy.eig(x)
    >>> w; v
    ivy.array(
        [
            [-3.72281323e-01+0.j,  5.37228132e+00+0.j],
            [3.88578059e-16+0.j,  1.10000000e+01+0.j]
        ]
    )
    ivy.array([
            [
                [-0.82456484+0.j, -0.41597356+0.j], [0.56576746+0.j, -0.90937671+0.j]
            ],
            [
                [-0.76822128+0.j, -0.70710678+0.j], [0.6401844 +0.j, -0.70710678+0.j]
            ]
    ])
    """
    return current_backend(x).eig(x)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_device
def eigvals(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> ivy.Array:
    """Compute eigenvalues of x. Returns a set of eigenvalues.

    Parameters
    ----------
    x
        An array of shape (..., N, N).

    Returns
    -------
    w
        Not necessarily ordered array(..., N) of eigenvalues in complex type.

    Examples
    --------
    With :class:`ivy.Array` inputs:
    >>> x = ivy.array([[1,2], [3,4]])
    >>> w = ivy.eigvals(x)
    >>> w
    ivy.array([-0.37228132+0.j,  5.37228132+0.j])

    >>> x = ivy.array([[[1,2], [3,4]], [[5,6], [5,6]]])
    >>> w = ivy.eigvals(x)
    >>> w
    ivy.array(
        [
            [-0.37228132+0.j,  5.37228132+0.j],
            [ 0.        +0.j, 11.        +0.j]
        ]
    )
    """
    return current_backend(x).eigvals(x)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def adjoint(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the complex conjugate transpose of x.

    Parameters
    ----------
    x
        An array with more than one dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        the complex conjugate transpose of the input.

    Examples
    --------
        >>> x = np.array([[1.-1.j, 2.+2.j],
                          [3.+3.j, 4.-4.j]])
        >>> x = ivy.array(x)
        >>> ivy.adjoint(x)
        ivy.array([[1.+1.j, 3.-3.j],
                   [2.-2.j, 4.+4.j]])
    """
    return current_backend(x).adjoint(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def lu_factor(
    A: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    pivot: bool = True,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Tuple[Union[ivy.Array, ivy.NativeArray], Union[ivy.Array, ivy.NativeArray]]:
    """
    Parameters
    ----------
    A
        tensor of shape (*, m, n) where * is zero or more batch dimensions.

    pivot
        Whether to compute the LU decomposition with partial pivoting, or the regular LU
        decomposition. pivot = False not supported on CPU. Default: True.

    out
        tuple of two tensors to write the output to. Ignored if None. Default: None.

    Returns
    -------
    ret
        A named tuple (LU, pivots).
    """
    return current_backend(A).lu_factor(A, pivot=pivot, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def lu_solve(
    lu: Union[ivy.Array, ivy.NativeArray],
    p: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return current_backend(lu, p, b).lu_solve(lu, p, b, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def solve_triangular(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    upper: bool = True,
    adjoint: bool = False,
    unit_diagonal: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the unique solution to the triangular system of linear equations
    AX = B.

    Parameters
    ----------
    x1
        Triangular coefficient array A of shape (..., N, N), with no zeros on diagonal.
    x2
        Right-hand side array B of shape (..., N, K).
    upper
        Whether the input `x1` is upper triangular.
    adjoint
        Whether to take the adjoint (conjugate transpose) of `x1` as the matrix A.
    unit_diagonal
        Whether to ignore the diagonal entries of A and assume them all equal to 1.
    out
        Optional output array. If provided, the output array to store the result.

    Returns
    -------
    ret
        The solution X, which has the same shape as B.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> a = ivy.array([[3, 0, 0, 0],
    ...                [2, 1, 0, 0],
    ...                [1, 0, 1, 0],
    ...                [1, 1, 1, 1]], dtype=ivy.float32)
    >>> b = ivy.array([[4],
    ...                [2],
    ...                [4],
    ...                [2]], dtype=ivy.float32)
    >>> x = ivy.solve_triangular(a, b, upper=False)
    >>> ivy.matmul(a, x)
    ivy.array([[4.],
               [2.],
               [4.],
               [2.]])
    """
    return current_backend(x1, x2).solve_triangular(
        x1, x2, upper=upper, adjoint=adjoint, unit_diagonal=unit_diagonal, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def multi_dot(
    x: Sequence[Union[ivy.Array, ivy.NativeArray]],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the dot product of two or more matrices in a single function
    call, while selecting the fastest evaluation order.

    Parameters
    ----------
    x
        sequence of matrices to multiply.
    out
        optional output array, for writing the result to. It must have a valid
        shape, i.e. the resulting shape after applying regular matrix multiplication
        to the inputs.

    Returns
    -------
    ret
        dot product of the arrays.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> A = ivy.arange(2 * 3).reshape((2, 3))
    >>> B = ivy.arange(3 * 2).reshape((3, 2))
    >>> C = ivy.arange(2 * 2).reshape((2, 2))
    >>> ivy.multi_dot((A, B, C))
    ivy.array([[ 26,  49],
               [ 80, 148]])

    >>> A = ivy.arange(2 * 3).reshape((2, 3))
    >>> B = ivy.arange(3 * 2).reshape((3, 2))
    >>> C = ivy.arange(2 * 2).reshape((2, 2))
    >>> D = ivy.zeros((2, 2))
    >>> ivy.multi_dot((A, B, C), out=D)
    >>> print(D)
    ivy.array([[ 26,  49],
               [ 80, 148]])
    """
    return current_backend(x).multi_dot(x, out=out)


multi_dot.mixed_backend_wrappers = {
    "to_add": ("handle_device",),
    "to_skip": (),
}


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def cond(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    p: Optional[Union[int, float, str]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the condition number of x.

    Parameters
    ----------
    x
        An array with more than one dimension.
    p
        The order of the norm of the matrix (see :func:`ivy.norm` for details).
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        the condition number of the input.

    Examples
    --------
    >>> x = ivy.array([[1., 2.],
    ...                [3., 4.]])
    >>> ivy.cond(x)
    ivy.array(14.933034)

    >>> x = ivy.array([[1., 2.],
    ...                     [3., 4.]])
    >>> ivy.cond(x, p=ivy.inf)
        ivy.array(21.0)
    """
    return current_backend(x).cond(x, p=p, out=out)


# This code has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/core_tenalg/_kronecker.py
@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def kronecker(
    x: Sequence[Union[ivy.Array, ivy.NativeArray]],
    skip_matrix: Optional[int] = None,
    reverse: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Kronecker product of a list of matrices.

    Parameters
    ----------
    x
        Sequence of matrices

    skip_matrix
        if not None, index of a matrix to skip

    reverse
        if True, the order of the matrices is reversed

    Returns
    -------
    kronecker_product: matrix of shape ``(prod(n_rows), prod(n_columns)``
        where ``prod(n_rows) = prod([m.shape[0] for m in matrices])``
        and ``prod(n_columns) = prod([m.shape[1] for m in matrices])``
    """
    if skip_matrix is not None:
        x = [x[i] for i in range(len(x)) if i != skip_matrix]

    if reverse:
        order = -1
    else:
        order = 1

    for i, matrix in enumerate(x[::order]):
        if not i:
            res = matrix
        else:
            res = ivy.kron(res, matrix, out=out)
    return res


# The code has been adapted from tensorly.khatri_rao
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/core_tenalg/_khatri_rao.py#L9
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def khatri_rao(
    x: Sequence[Union[ivy.Array, ivy.NativeArray]],
    weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    skip_matrix: Optional[Sequence[int]] = None,
    mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Khatri-Rao product of a sequence of matrices.

        This can be seen as a column-wise kronecker product.
        If one matrix only is given, that matrix is directly returned.

    Parameters
    ----------
    x
        Sequence of tensors with the same number of columns, i.e.::
            for i in len(x):
                x[i].shape = (n_i, m)

    weights
        array of weights for each rank, of length m, the number of column of the factors
        (i.e. m == factor[i].shape[1] for any factor)

    skip_matrix
        if not None, index of a matrix to skip

    mask
        array of 1s and 0s of length m

    out
        optional output array, for writing the result to. It must have a shape that the
        result can broadcast to.

    Returns
    -------
    khatri_rao_product: ivy.Array of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in input])``
        i.e. the product of the number of rows of all the input in the product.
    """
    if skip_matrix is not None:
        x = [x[i] for i in range(len(x)) if i != skip_matrix]

    # Khatri-rao of only one matrix: just return that matrix
    if len(x) == 1:
        if ivy.exists(out):
            return ivy.inplace_update(out, x[0])
        return x[0]

    if len(x[0].shape) == 2:
        n_columns = x[0].shape[1]
    else:
        n_columns = 1
        x = [ivy.reshape(m, (-1, 1)) for m in x]
        logging.warning(
            "Khatri-rao of a series of vectors instead of input. "
            "Considering each as a matrix with 1 column."
        )

    # Testing whether the input have the proper size
    for i, matrix in enumerate(x):
        if len(matrix.shape) != 2:
            raise ValueError(
                "All the input must have exactly 2 dimensions!"
                f"Matrix {i} has dimension {len(matrix.shape)} != 2."
            )
        if matrix.shape[1] != n_columns:
            raise ValueError(
                "All input must have same number of columns!"
                f"Matrix {i} has {matrix.shape[1]} columns != {n_columns}."
            )

    for i, e in enumerate(x[1:]):
        if not i:
            if weights is None:
                res = x[0]
            else:
                res = x[0] * ivy.reshape(weights, (1, -1))
        s1, s2 = ivy.shape(res)
        s3, s4 = ivy.shape(e)

        a = ivy.reshape(res, (s1, 1, s2))
        b = ivy.reshape(e, (1, s3, s4))
        res = ivy.reshape(a * b, (-1, n_columns))

    m = ivy.reshape(mask, (1, -1)) if mask is not None else 1

    res = res * m

    if ivy.exists(out):
        return ivy.inplace_update(out, res)

    return res


# The following code has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/core_tenalg/n_mode_product.py#L5
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def mode_dot(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    matrix_or_vector: Union[ivy.Array, ivy.NativeArray],
    mode: int,
    transpose: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """N-mode product of a tensor and a matrix or vector at the specified mode.

    Parameters
    ----------
    x
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode
        int in the range(1, N)
    transpose
        If True, the matrix is transposed.
        For complex tensors, the conjugate transpose is used.
    out
        optional output array, for writing the result to. It must have a shape that the
        result can broadcast to.

    Returns
    -------
    ivy.Array
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
          if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`
          if matrix_or_vector is a vector
    """
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(x.shape)
    ndims = len(matrix_or_vector.shape)

    if ndims == 2:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != x.shape[mode]:
            raise ValueError(
                f"shapes {x.shape} and {matrix_or_vector.shape} not aligned in"
                f" mode-{mode} multiplication: {x.shape[mode]} (mode {mode}) !="
                f" {matrix_or_vector.shape[dim]} (dim 1 of matrix)"
            )

        if transpose:
            matrix_or_vector = ivy.conj(ivy.permute_dims(matrix_or_vector, (1, 0)))

        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif ndims == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != x.shape[mode]:
            raise ValueError(
                f"shapes {x.shape} and {matrix_or_vector.shape} not aligned for"
                f" mode-{mode} multiplication: {x.shape[mode]} (mode {mode}) !="
                f" {matrix_or_vector.shape[0]} (vector size)"
            )
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            new_shape = []
        vec = True

    else:
        raise ValueError(
            "Can only take n_mode_product with a vector or a matrix."
            f"Provided array of dimension {ndims} not in [1, 2]."
        )

    res = ivy.matmul(matrix_or_vector, ivy.unfold(x, mode))

    if vec:  # We contracted with a vector, leading to a vector
        return ivy.reshape(res, new_shape, out=out)
    else:  # tensor times vec: refold the unfolding
        return ivy.fold(res, fold_mode, new_shape, out=out)


# The following code has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/core_tenalg/n_mode_product.py#L81
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def multi_mode_dot(
    x: Union[ivy.Array, ivy.NativeArray],
    mat_or_vec_list: Sequence[Union[ivy.Array, ivy.NativeArray]],
    /,
    modes: Optional[Sequence[int]] = None,
    skip: Optional[Sequence[int]] = None,
    transpose: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""N-mode product of a tensor and several matrices or vectors over several
    modes.

    Parameters
    ----------
    x
        the input tensor

    mat_or_vec_list
         sequence of matrices or vectors of length ``tensor.ndim``

    skip
        None or int, optional, default is None
        If not None, index of a matrix to skip.

    modes
        None or int list, optional, default is None

    transpose
        If True, the matrices or vectors in in the list are transposed.
        For complex tensors, the conjugate transpose is used.
    out
        optional output array, for writing the result to. It must have a shape that the
        result can broadcast to.

    Returns
    -------
    ivy.Array
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:
    :math:`\\text{x  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`
    """  # noqa: E501
    if modes is None:
        modes = range(len(mat_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = x

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(mat_or_vec_list, modes), key=lambda x: x[1])
    for i, (mat_or_vec_list, mode) in enumerate(factors_modes):
        ndims = len(mat_or_vec_list.shape)
        if (skip is not None) and (i == skip):
            continue

        if transpose and ndims == 2:
            res = mode_dot(
                res,
                ivy.conj(ivy.permute_dims(mat_or_vec_list, (1, 0))),
                mode - decrement,
            )
        else:
            res = mode_dot(res, mat_or_vec_list, mode - decrement)

        if ndims == 1:
            decrement += 1

    if ivy.exists(out):
        return ivy.inplace_update(out, res)

    return res


def _svd_checks(x, n_eigenvecs=None):
    """Run common checks to all of the SVD methods.

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    n_eigenvecs : int
        the number of eigenvectors to solve for
    min_dim : int
        the minimum dimension of matrix
    max_dim : int
        the maximum dimension of matrix
    """
    # ndims = len(x.shape)
    # if ndims != 2:
    #    raise ValueError(f"matrix be a matrix. matrix.ndim is {ndims} != 2")

    dim_1, dim_2 = ivy.shape(x)[-2:]
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if n_eigenvecs > max_dim:
        logging.warning(
            f"Trying to compute SVD with n_eigenvecs={n_eigenvecs}, which is larger "
            f"than max(matrix.shape)={max_dim}. Setting n_eigenvecs to {max_dim}."
        )
        n_eigenvecs = max_dim

    return n_eigenvecs, min_dim, max_dim


# This function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/svd.py#L12
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def svd_flip(
    U: Union[ivy.Array, ivy.NativeArray],
    V: Union[ivy.Array, ivy.NativeArray],
    /,
    u_based_decision: Optional[bool] = True,
) -> Tuple[ivy.Array, ivy.Array]:
    """Sign correction to ensure deterministic output from SVD. Adjusts the
    columns of u and the rows of v such that the loadings in the columns in u
    that are largest in absolute value are always positive. This function is
    borrowed from scikit- learn/utils/extmath.py.

    Parameters
    ----------
    U
        left singular matrix output of SVD
    V
        right singular matrix output of SVD
    u_based_decision
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of U, rows of V
        max_abs_cols = ivy.argmax(ivy.abs(U), axis=0)
        signs = ivy.sign(
            ivy.array(
                [U[i, j] for (i, j) in zip(max_abs_cols, range(ivy.shape(U)[1]))],
            )
        )
        U = U * signs
        if ivy.shape(V)[0] > ivy.shape(U)[1]:
            signs = ivy.concat((signs, ivy.ones(ivy.shape(V)[0] - ivy.shape(U)[1])))
        V = V * signs[: ivy.shape(V)[0]][:, None]
    else:
        # rows of V, columns of U
        max_abs_rows = ivy.argmax(ivy.abs(V), axis=1)
        signs = ivy.sign(
            ivy.array(
                [V[i, j] for (i, j) in zip(range(ivy.shape(V)[0]), max_abs_rows)],
            )
        )
        V = V * signs[:, None]
        if ivy.shape(U)[1] > ivy.shape(V)[0]:
            signs = ivy.concat(
                (
                    signs,
                    ivy.ones(
                        ivy.shape(U)[1] - ivy.shape(V)[0],
                    ),
                )
            )
        U = U * signs[: ivy.shape(U)[1]]

    return U, V


# This function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/svd.py#L65
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def make_svd_non_negative(
    x: Union[ivy.Array, ivy.NativeArray],
    U: Union[ivy.Array, ivy.NativeArray],
    S: Union[ivy.Array, ivy.NativeArray],
    V: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    nntype: Optional[Literal["nndsvd", "nndsvda"]] = "nndsvd",
) -> Tuple[ivy.Array, ivy.Array]:
    """Use NNDSVD method to transform SVD results into a non-negative form.
    This method leads to more efficient solving with NNMF [1].

    Parameters
    ----------
    x
        tensor being decomposed.
    U
        left singular matrix from SVD.
    S
        diagonal matrix from SVD.
    V
        right singular matrix from SVD.
    nntype
        whether to fill small values with 0.0 (nndsvd),
          or the tensor mean (nndsvda, default).

    [1]: Boutsidis & Gallopoulos. Pattern Recognition, 41(4): 1350-1362, 2008.
    """
    W = ivy.zeros_like(U)
    H = ivy.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = ivy.sqrt(S[0]) * ivy.abs(U[:, 0])
    H[0, :] = ivy.sqrt(S[0]) * ivy.abs(V[0, :])

    for j in range(1, len(S)):
        a, b = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        a_p, b_p = ivy.where(a < 0.0, 0, a), ivy.where(b < 0.0, 0.0, b)
        # a_p, b_p = ivy.clip(a, 0.0), ivy.clip(b, 0.0)
        # a_n, b_n = ivy.abs(ivy.clip(a, 0.0)), ivy.abs(ivy.clip(b, 0.0))
        a_n, b_n = ivy.abs(ivy.where(a > 0.0, 0.0, a)), ivy.abs(
            ivy.where(b > 0.0, 0.0, b)
        )

        # and their norms
        a_p_nrm, b_p_nrm = float(ivy.vector_norm(a_p)), float(ivy.vector_norm(b_p))
        a_n_nrm, b_n_nrm = float(ivy.vector_norm(a_n)), float(ivy.vector_norm(b_n))

        m_p, m_n = a_p_nrm * b_p_nrm, a_n_nrm * b_n_nrm

        # choose update
        if m_p > m_n:
            u = a_p / a_p_nrm
            v = b_p / b_p_nrm
            sigma = m_p
        else:
            u = a_n / a_n_nrm
            v = b_n / b_n_nrm
            sigma = m_n

        lbd = float(ivy.sqrt(S[j] * sigma))
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    # After this point we no longer need H
    eps = ivy.finfo(x.dtype).min

    if nntype == "nndsvd":
        W = ivy.soft_thresholding(W, eps)
        H = ivy.soft_thresholding(H, eps)
    elif nntype == "nndsvda":
        avg = ivy.mean(x)
        W = ivy.where(eps > W, ivy.ones(ivy.shape(W)) * avg, W)
        H = ivy.where(eps > H, ivy.ones(ivy.shape(H)) * avg, H)
    else:
        raise ValueError(
            f'Invalid nntype parameter: got {nntype} instead of one of ("nndsvd",'
            ' "nndsvda")'
        )

    return W, H


# The following function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/svd.py#L206
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def truncated_svd(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    compute_uv: bool = True,
    n_eigenvecs: Optional[int] = None,
) -> Union[ivy.Array, Tuple[ivy.Array, ivy.Array, ivy.Array]]:
    """Compute a truncated SVD on `x` using the standard SVD.

    Parameters
    ----------
    x
        2D-array
        compute_uv
        If ``True`` then left and right singular vectors will be computed and returned
        in ``U`` and ``Vh``, respectively. Otherwise, only the singular values will be
        computed, which can be significantly faster.
    n_eigenvecs
        if specified, number of eigen[vectors-values] to return
        else full matrices will be returned

    Returns
    -------
    ret
        a namedtuple ``(U, S, Vh)``
        Each returned array must have the same floating-point data type as ``x``.
    """
    n_eigenvecs, min_dim, _ = _svd_checks(x, n_eigenvecs=n_eigenvecs)
    full_matrices = True if n_eigenvecs > min_dim else False

    if compute_uv:
        U, S, Vh = ivy.svd(x, full_matrices=full_matrices, compute_uv=True)
        return U[:, :n_eigenvecs], S[:n_eigenvecs], Vh[:n_eigenvecs, :]
    else:
        S = ivy.svd(x, full_matrices=full_matrices, compute_uv=False)
        return S[:n_eigenvecs]


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def tensor_train(
    input_tensor: Union[ivy.Array, ivy.NativeArray],
    rank: Union[int, Sequence[int]],
    /,
    *,
    svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
    verbose: Optional[bool] = False,
) -> ivy.TTTensor:
    """TT decomposition via recursive SVD.

    Decomposes the input into a sequence of order-3 tensors (factors)
    Also known as Tensor-Train decomposition [1]_

    Parameters
    ----------
    input_tensor
        tensor to decompose
    rank
        maximum allowable TT rank of the factors
        if int, then this is the same for all the factors
        if int list, then rank[k] is the rank of the kth factor
    svd
        function to use to compute the SVD
    verbose
        level of verbosity

    Returns
    -------
    factors
        order-3 tensors of the TT decomposition

    [1]: Ivan V. Oseledets. "Tensor-train decomposition",
    SIAM J. Scientific Computing, 33(5):2295–2317, 2011.
    """
    rank = ivy.TTTensor.validate_tt_rank(ivy.shape(input_tensor), rank=rank)
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    unfolding = input_tensor
    factors = [None] * n_dim

    for k in range(n_dim - 1):
        n_row = int(rank[k] * tensor_size[k])
        unfolding = ivy.reshape(unfolding, (n_row, -1))

        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        U, S, V = _svd_interface(unfolding, n_eigenvecs=current_rank, method=svd)

        rank[k + 1] = current_rank
        factors[k] = ivy.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        if verbose is True:
            print(
                "TT factor " + str(k) + " computed with shape " + str(factors[k].shape)
            )

        unfolding = ivy.reshape(S, (-1, 1)) * V

    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = ivy.reshape(unfolding, (prev_rank, last_dim, 1))

    if verbose is True:
        print(
            "TT factor "
            + str(n_dim - 1)
            + " computed with shape "
            + str(factors[n_dim - 1].shape)
        )

    return ivy.TTTensor(factors)


# TODO uncomment the code below when these svd
# methods have been added
def _svd_interface(
    matrix,
    method="truncated_svd",
    n_eigenvecs=None,
    flip_sign=True,
    u_based_flip_sign=True,
    non_negative=None,
    mask=None,
    n_iter_mask_imputation=5,
    **kwargs,
):
    if method == "truncated_svd":
        svd_fun = truncated_svd
    # elif method == "symeig_svd":
    #    svd_fun = symeig_svd
    # elif method == "randomized_svd":
    #    svd_fun = randomized_svd
    elif callable(method):
        svd_fun = method
    else:
        raise ValueError("Invalid Choice")

    U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)
    if mask is not None and n_eigenvecs is not None:
        for _ in range(n_iter_mask_imputation):
            S = S * ivy.eye(U.shape[-1], V.shape[-2])
            matrix = matrix * mask + (U @ S @ V) * (1 - mask)
            U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if flip_sign:
        U, V = svd_flip(U, V, u_based_decision=u_based_flip_sign)

    if non_negative is not False and non_negative is not None:
        U, V = make_svd_non_negative(matrix, U, S, V)

    return U, S, V


# This function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_tucker.py#L22


# TODO update svd type hints when other svd methods have been added
# also update the test
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def initialize_tucker(
    x: Union[ivy.Array, ivy.NativeArray],
    rank: Sequence[int],
    modes: Sequence[int],
    /,
    *,
    init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
    seed: Optional[int] = None,
    svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
    non_negative: Optional[bool] = False,
    mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    svd_mask_repeats: Optional[int] = 5,
) -> Tuple[ivy.Array, Sequence[ivy.Array]]:
    """Initialize core and factors used in `tucker`. The type of initialization
    is set using `init`. If `init == 'random'` then initialize factor matrices
    using `random_state`. If `init == 'svd'` then initialize the `m`th factor
    matrix using the `rank` left singular vectors of the `m`th unfolding of the
    input tensor.

    Parameters
    ----------
    x
        input tensor
    rank
        number of components
    modes
        modes to consider in the input tensor
    seed
        Used to create a random seed distribution
        when init == 'random'
    init
        initialization scheme for tucker decomposition.
    svd
        function to use to compute the SVD
    non_negative
        if True, non-negative factors are returned
    mask
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
    svd_mask_repeats
        number of iterations for imputing the values in the SVD matrix when
        mask is not None

    Returns
    -------
    core
        initialized core tensor
    factors
        list of factors
    """
    try:
        assert len(x.shape) >= 2
    except ValueError as e:
        raise ValueError(
            "expected x to have at least 2 dimensions but it has only"
            f" {len(x.shape)} dimension(s)"
        ) from e

    # Initialisation
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            mask_unfold = None if mask is None else ivy.unfold(mask, mode)
            U, _, _ = _svd_interface(
                ivy.unfold(x, mode),
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                # random_state=random_state,
            )
            factors.append(U)

        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(x, factors, modes=modes, transpose=True)

    elif init == "random":
        core = (
            ivy.random_uniform(
                shape=[rank[index] for index in range(len(modes))],
                dtype=x.dtype,
                seed=seed,
            )
            + 0.01
        )
        factors = [
            ivy.random_uniform(
                shape=(x.shape[mode], rank[index]), dtype=x.dtype, seed=seed
            )
            for index, mode in enumerate(modes)
        ]

    else:
        (core, factors) = init

    if non_negative is True:
        factors = [ivy.abs(f) for f in factors]
        core = ivy.abs(core)

    return (core, factors)


# This function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/decomposition/_tucker.py#L98
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def partial_tucker(
    x: Union[ivy.Array, ivy.NativeArray],
    rank: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[int]] = None,
    /,
    *,
    n_iter_max: Optional[int] = 100,
    init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
    svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
    seed: Optional[int] = None,
    mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    svd_mask_repeats: Optional[int] = 5,
    tol: Optional[float] = 10e-5,
    verbose: Optional[bool] = False,
    return_errors: Optional[bool] = False,
) -> Tuple[ivy.Array, Sequence[ivy.Array]]:
    """Partial tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition
        exclusively along the provided modes.

    Parameters
    ----------
    x
        the  input tensor
    rank
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
        if None, original tensors size will be preserved.
    modes
        list of the modes on which to perform the decomposition
    n_iter_max
        maximum number of iteration
    init
        {'svd', 'random'}, or TuckerTensor optional
        if a TuckerTensor is provided, this is used for initialization
    svd
        str, default is 'truncated_svd'
        function to use to compute the SVD,
    seed
        Used to create a random seed distribution
        when init == 'random'
    mask
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
    svd_mask_repeats
        number of iterations for imputing the values in the SVD matrix when
        mask is not None
    tol
        tolerance: the algorithm stops when the variation in
        the reconstruction error is less than the tolerance.
    verbose
        if True, different in reconstruction errors are returned at each
        iteration.
    return_erros
        if True, list of reconstruction errors are returned.

    Returns
    -------
    core : ndarray
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``
    """
    if modes is None:
        modes = list(range(len(x.shape)))

    if rank is None:
        logging.warning(
            "No value given for 'rank'. The decomposition will preserve the original"
            " size."
        )
        rank = [ivy.shape(x)[mode] for mode in modes]
    elif isinstance(rank, int):
        logging.warning(
            f"Given only one int for 'rank' instead of a list of {len(modes)} modes."
            " Using this rank for all modes."
        )
        rank = tuple(rank for _ in modes)
    else:
        rank = ivy.TuckerTensor.validate_tucker_rank(x.shape, rank=rank)

    # SVD init
    core, factors = initialize_tucker(
        x,
        rank,
        modes,
        init=init,
        svd=svd,
        seed=seed,
        mask=mask,
        svd_mask_repeats=svd_mask_repeats,
    )

    rec_errors = []
    norm_tensor = ivy.sqrt(ivy.sum(x**2))

    for iteration in range(n_iter_max):
        if mask is not None:
            x = x * mask + multi_mode_dot(
                core, factors, modes=modes, transpose=False
            ) * (1 - mask)

        for index, mode in enumerate(modes):
            core_approximation = multi_mode_dot(
                x, factors, modes=modes, skip=index, transpose=True
            )
            eigenvecs, _, _ = _svd_interface(
                ivy.unfold(core_approximation, mode),
                n_eigenvecs=rank[index],
                # random_state=random_state,
            )
            factors[index] = eigenvecs

        core = multi_mode_dot(x, factors, modes=modes, transpose=True)

        # The factors are orthonormal and
        #  therefore do not affect the reconstructed tensor's norm
        norm_core = ivy.sqrt(ivy.sum(core**2))
        rec_error = ivy.sqrt(abs(norm_tensor**2 - norm_core**2)) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(
                    f"reconstruction error={rec_errors[-1]},"
                    f" variation={rec_errors[-2] - rec_errors[-1]}."
                )

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print(f"converged in {iteration} iterations.")
                break

    if return_errors:
        return (core, factors), rec_errors
    return (core, factors)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def tucker(
    x: Union[ivy.Array, ivy.NativeArray],
    rank: Optional[Sequence[int]] = None,
    /,
    *,
    fixed_factors: Optional[Sequence[int]] = None,
    n_iter_max: Optional[int] = 100,
    init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
    svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
    seed: Optional[int] = None,
    mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    svd_mask_repeats: Optional[int] = 5,
    tol: Optional[float] = 10e-5,
    verbose: Optional[bool] = False,
    return_errors: Optional[bool] = False,
):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    x
        input tensor
    rank
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    fixed_factors
        if not None, list of modes for which to keep the factors fixed.
        Only valid if a Tucker tensor is provided as init.
    n_iter_max
        maximum number of iteration
    init
        {'svd', 'random'}, or TuckerTensor optional
        if a TuckerTensor is provided, this is used for initialization
    svd
        str, default is 'truncated_svd'
        function to use to compute the SVD,
    seed
        Used to create a random seed distribution
        when init == 'random'
    mask
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
    svd_mask_repeats
        number of iterations for imputing the values in the SVD matrix when
        mask is not None
    tol
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    verbose
        if True, different in reconstruction errors are returned at each
        iteration.

    return_errors
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False

    Returns
    -------
        ivy.TuckerTensor or ivy.TuckerTensor and
        list of reconstruction errors if return_erros is True.

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if fixed_factors:
        try:
            (core, factors) = init
        except ValueError as e:
            raise ValueError(
                f"Got fixed_factor={fixed_factors} but no appropriate Tucker tensor was"
                ' passed for "init".'
            ) from e
        if len(fixed_factors) == len(factors):
            return ivy.TuckerTensor((core, factors))

        fixed_factors = sorted(fixed_factors)
        modes_fixed, factors_fixed = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i in fixed_factors]
        )
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed)
        modes, factors = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i not in fixed_factors]
        )
        init = (core, list(factors))

        rank = ivy.TuckerTensor.validate_tucker_rank(x.shape, rank=rank)
        (core, new_factors), rec_errors = partial_tucker(
            x,
            rank,
            modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            seed=seed,
            mask=mask,
            verbose=verbose,
            svd_mask_repeats=svd_mask_repeats,
            return_errors=True,
        )

        factors = list(new_factors)
        for i, e in enumerate(fixed_factors):
            factors.insert(e, factors_fixed[i])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed, transpose=True)

        if return_errors:
            return ivy.TuckerTensor((core, factors)), rec_errors
        return ivy.TuckerTensor((core, factors))

    else:
        modes = list(range(len(x.shape)))
        rank = ivy.TuckerTensor.validate_tucker_rank(x.shape, rank=rank)

        (core, factors), rec_errors = partial_tucker(
            x,
            rank,
            modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            seed=seed,
            mask=mask,
            verbose=verbose,
            return_errors=True,
        )
        if return_errors:
            return ivy.TuckerTensor((core, factors)), rec_errors
        else:
            return ivy.TuckerTensor((core, factors))


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def tt_matrix_to_tensor(
    tt_matrix: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the full tensor whose TT-Matrix decomposition is given by
    'factors' Re- assembles 'factors', which represent a tensor in TT-Matrix
    format into the corresponding full tensor.

    Parameters
    ----------
    tt_matrix
            array of 4D-arrays
            TT-Matrix factors (known as core) of shape
            (rank_k, left_dim_k, right_dim_k, rank_{k+1})

    out
        Optional output array. If provided, the output array to store the result.

    Returns
    -------
    output_tensor: array
                tensor whose TT-Matrix decomposition was given by 'factors'

    Examples
    --------
    >>> x = ivy.array([[[[[0.49671414],
    ...                      [-0.1382643]],
    ...
    ...                     [[0.64768857],
    ...                      [1.5230298]]]],
    ...                   [[[[-0.23415337],
    ...                      [-0.23413695]],
    ...
    ...                     [[1.57921278],
    ...                      [0.76743472]]]]])
    >>> y = ivy.tt_matrix_to_tensor(x)
    >>> print(y)
    ivy.array([[[[-0.1163073 , -0.11629914],
    [ 0.03237505,  0.03237278]],

    [[ 0.78441733,  0.38119566],
    [-0.21834874, -0.10610882]]],


    [[[-0.15165846, -0.15164782],
    [-0.35662258, -0.35659757]],

    [[ 1.02283812,  0.49705869],
    [ 2.40518808,  1.16882598]]]])
    """
    _, in_shape, out_shape, _ = zip(*(f.shape for f in tt_matrix))
    ndim = len(in_shape)
    full_shape = sum(zip(*(in_shape, out_shape)), ())
    order = list(range(0, ndim * 2, 2)) + list(range(1, ndim * 2, 2))
    for i, factor in enumerate(tt_matrix):
        if not i:
            res = factor
        else:
            res = ivy.tensordot(res, factor, axes=([len(res.shape) - 1], [0]))
    return ivy.permute_dims(ivy.reshape(res, full_shape), axes=order, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def dot(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the dot product between two arrays `a` and `b` using the current
    backend's implementation. The dot product is defined as the sum of the
    element-wise product of the input arrays.

    Parameters
    ----------
    a
        First input array.
    b
        Second input array.
    out
        Optional output array. If provided, the output array to store the result.

    Returns
    -------
    ret
        The dot product of the input arrays.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> a = ivy.array([1, 2, 3])
    >>> b = ivy.array([4, 5, 6])
    >>> result = ivy.dot(a, b)
    >>> print(result)
    ivy.array(32)

    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> b = ivy.array([[5, 6], [7, 8]])
    >>> c = ivy.empty_like(a)
    >>> ivy.dot(a, b, out=c)
    >>> print(c)
    ivy.array([[19, 22],
           [43, 50]])

    >>> a = ivy.array([[1.1, 2.3, -3.6]])
    >>> b = ivy.array([[-4.8], [5.2], [6.1]])
    >>> c = ivy.zeros((1, 1))
    >>> ivy.dot(a, b, out=c)
    >>> print(c)
    ivy.array([[-15.28]])
    """
    return current_backend(a, b).dot(a, b, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def general_inner_product(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    n_modes: Optional[int] = None,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Generalised inner products between tensors.

        Takes the inner product between the last (respectively first)
        `n_modes` of `a` (respectively `b`)

    Parameters
    ----------
    a
        first input tensor.
    b
        second input tensor.
    n_modes
        int, default is None. If None, the traditional inner product is returned
        (i.e. a float) otherwise, the product between the `n_modes` last modes of
        `a` and the `n_modes` first modes of `b` is returned. The resulting tensor's
        order is `len(a) - n_modes`.
    out
        Optional output array. If provided, the output array to store the result.

    Returns
    -------
        The inner product of the input arrays.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> a = ivy.array([1, 2, 3])
    >>> b = ivy.array([4, 5, 6])
    >>> result = ivy.general_inner_product(a, b, 1)
    >>> print(result)
    ivy.array(32)

    >>> a = ivy.array([1, 2])
    >>> b = ivy.array([4, 5])
    >>> result = ivy.general_inner_product(a, b)
    >>> print(result)
    ivy.array(14)

    >>> a = ivy.array([[1, 1], [1, 1]])
    >>> b = ivy.array([[1, 2, 3, 4],[1, 1, 1, 1]])
    >>> result = ivy.general_inner_product(a, b, 1)
    >>> print(result)
    ivy.array([[2, 3, 4, 5],
       [2, 3, 4, 5]])
    """
    shape_a = a.shape
    shape_b = b.shape
    if n_modes is None:
        if shape_a != shape_b:
            raise ValueError(
                "Taking a generalised product between two tensors without specifying"
                " common modes is equivalent to taking inner product.This requires"
                f" a.shape == b.shape.However, got shapes {a.shape} and {b.shape}"
            )
        return ivy.sum(ivy.multiply(a, b), out=out)

    common_modes = shape_a[len(shape_a) - n_modes :]
    if common_modes != shape_b[:n_modes]:
        raise ValueError(
            f"Incorrect shapes for inner product along {n_modes} common modes."
            f"Shapes {shape_a.shape} and {shape_b.shape}"
        )

    common_size = int(ivy.prod(common_modes)) if len(common_modes) != 0 else 0
    output_shape = shape_a[:-n_modes] + shape_b[n_modes:]
    inner_product = ivy.dot(
        ivy.reshape(a, (-1, common_size)), ivy.reshape(b, (common_size, -1))
    )
    return ivy.reshape(inner_product, output_shape, out=out)


# This function has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tenalg/core_tenalg/moments.py#L5


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def higher_order_moment(
    x: Union[ivy.Array, ivy.NativeArray],
    order: int,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Higher-Order Moment.

    Parameters
    ----------
    x
        matrix of size (n_samples, n_features)
        or tensor of size(n_samples, D1, ..., DN)

    order
        number of the higher-order moment to compute

    Returns
    -------
    tensor
        if tensor is a matrix of size (n_samples, n_features),
        tensor of size (n_features, )*order

    Examples
    --------
    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> result = ivy.higher_order_moment(a, 3)
    >>> print(result)
    ivy.array([[
        [14, 19],
        [19, 26]],
       [[19, 26],
        [26, 36]
    ]])
    """
    moment = ivy.copy_array(x)
    for _ in range(order - 1):
        moment = ivy.batched_outer([moment, x])
    return ivy.mean(moment, axis=0, out=out)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device
def batched_outer(
    tensors: Sequence[Union[ivy.Array, ivy.NativeArray]],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a generalized outer product of the tensors.

    Parameters
    ----------
    tensors
        list of tensors of shape (n_samples, J1, ..., JN) ,
        (n_samples, K1, ..., KM) ...

    Returns
    -------
    outer product of tensors
        of shape (n_samples, J1, ..., JN, K1, ..., KM, ...)

    Examples
    --------
    >>> a = ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> b = ivy.array([[[.1, .2], [.3, .4]], [[.5, .6], [.7, .8]]])
    >>> result = ivy.batched_outer([a, b])
    >>> print(result)
    ivy.array([[[[[0.1, 0.2],
          [0.30000001, 0.40000001]],
         [[0.2       , 0.40000001],
          [0.60000002, 0.80000001]]],
        [[[0.3       , 0.60000001],
          [0.90000004, 1.20000002]],
         [[0.40000001, 0.80000001],
          [1.20000005, 1.60000002]]]],
       [[[[2.5       , 3.00000012],
          [3.49999994, 4.00000006]],
         [[3.        , 3.60000014],
          [4.19999993, 4.80000007]]],
        [[[3.5       , 4.20000017],
          [4.89999992, 5.60000008]],
         [[4.        , 4.80000019],
          [5.5999999 , 6.4000001 ]]]]])
    """
    result = None
    result_size = None
    result_shape = None
    for i, tensor in enumerate(tensors):
        if i:
            current_shape = ivy.shape(tensor)
            current_size = len(current_shape) - 1

            n_samples = current_shape[0]

            _check_same_batch_size(i, n_samples, result_shape)

            shape_1 = result_shape + (1,) * current_size
            shape_2 = (n_samples,) + (1,) * result_size + tuple(current_shape[1:])

            result = ivy.reshape(result, shape_1) * ivy.reshape(tensor, shape_2)
        else:
            result = tensor

        result_shape = ivy.shape(result)
        result_size = len(result_shape) - 1

    if ivy.exists(out):
        result = ivy.inplace_update(out, result)

    return result


def _check_same_batch_size(i, n_samples, result_shape):
    if n_samples != result_shape[0]:
        raise ValueError(
            f"Tensor {i} has a batch-size of {n_samples} but those before had a"
            f" batch-size of {result_shape[0]}, all tensors should have the"
            " same batch-size."
        )
