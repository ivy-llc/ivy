# global
from typing import Union, Optional, Tuple, List, NamedTuple

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.exceptions import handle_exceptions

# Helpers #
# ------- #


def _check_valid_dimension_size(std):
    ivy.assertions.check_dimensions(std)


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def eigh_tridiagonal(
    alpha: Union[ivy.Array, ivy.NativeArray],
    beta: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    eigvals_only: bool = True,
    select: str = 'a',
    select_range: Optional[Union[Tuple[int], List[int], ivy.Array, ivy.NativeArray]] = None,
    tol: Optional[float] = None,
) -> Union[ivy.Array, Tuple[ivy.Array]]:
    """
    """
    if ivy.current_backend(alpha).backend == "tensorflow":
        return ivy.current_backend(alpha).eigh_tridiagonal(
            alpha,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol
        )
    x = ivy.diag(alpha)
    y = ivy.diag(beta, k=1)
    z = ivy.diag(beta, k=-1)
    w = x+y+z

    eigh_out = ivy.linalg.eigh(w)
    eigenvalues = eigh_out.eigenvalues
    eigenvectors = eigh_out.eigenvectors

    if select == 'i':
        eigenvalues = eigenvalues[select_range[0]:select_range[1]+1]
        eigenvectors = eigenvectors[:,select_range[0]:select_range[1]+1]
    elif select == 'v':
        condition = ivy.logical_and(
            eigenvalues.greater(select_range[0]),
            eigenvalues.less_equal(select_range[1])
        )
        eigenvalues = eigenvalues[condition]
        eigenvectors = eigenvectors[:,condition]

    if eigvals_only:
        return eigenvalues

    result_tuple = NamedTuple(
        "eigh",
        [
            ("eigenvalues", ivy.NativeArray),
            ("eigenvectors", ivy.NativeArray),
        ],
    )
    return result_tuple(eigenvalues, eigenvectors)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def diagflat(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = -1,
    num_cols: Optional[int] = -1,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Returns a two-dimensional array with the flattened input as a diagonal.

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


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/extensions/generated/signatures.linalg.diagonal.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def kron(
    a: Union[ivy.Array, ivy.NativeArray],
    b: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Kronecker product, a composite array
    made of blocks of the second array scaled by the first.

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def matrix_exp(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the matrix exponential of a square matrix.

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


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def eig(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple[ivy.Array]:
    """Computes eigenvalies and eigenvectors of x. Returns a tuple with two elements:
     first is the set of eigenvalues, second is the set of eigenvectors.

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


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def eigvals(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
) -> ivy.Array:
    """Computes eigenvalues of x. Returns a set of eigenvalues.

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def adjoint(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the complex conjugate transpose of x.

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
