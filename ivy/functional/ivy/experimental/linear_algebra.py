# global
from typing import Union, Optional, Tuple, List, Sequence

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_array_function,
    handle_device_shifting,
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
    """
    Compute the eigenvalues and eigenvectors of a Hermitian tridiagonal matrix.

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
@handle_device_shifting
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
    """
    Return a two-dimensional array with the flattened input as a diagonal.

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

    Functional Examples
    ------------------

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
@handle_device_shifting
def kron(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the Kronecker product, a composite array made of blocks of the second array
    scaled by the first.

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
@handle_device_shifting
def matrix_exp(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the matrix exponential of a square matrix.

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
@handle_device_shifting
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

    Functional Examples
    ------------------
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
@handle_device_shifting
def eigvals(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> ivy.Array:
    """
    Compute eigenvalues of x. Returns a set of eigenvalues.

    Parameters
    ----------
    x
        An array of shape (..., N, N).

    Returns
    -------
    w
        Not necessarily ordered array(..., N) of eigenvalues in complex type.

    Functional Examples
    ------------------
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
@handle_device_shifting
def adjoint(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the complex conjugate transpose of x.

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
@handle_out_argument
@to_native_arrays_and_back
def multi_dot(
    x: Sequence[Union[ivy.Array, ivy.NativeArray]],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the dot product of two or more matrices in a single function call, while
    selecting the fastest evaluation order.

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
    "to_add": ("handle_device_shifting",),
    "to_skip": (),
}


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def cond(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    p: Optional[Union[int, float, str]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the condition number of x.

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
                           [3., 4.]])
        >>> ivy.cond(x)
        ivy.array(14.933034)

        >>> x = ivy.array([[1., 2.],
                            [3., 4.]])
        >>> ivy.cond(x, p=ivy.inf)
        ivy.array(21.0)
    """
    return current_backend(x).cond(x, p=p, out=out)
