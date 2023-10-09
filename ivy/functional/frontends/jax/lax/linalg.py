import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


@to_ivy_arrays_and_back
def eig(x, /, *, compute_left_eigenvectors=True, compute_right_eigenvectors=True):
    return ivy.eig(x)


@to_ivy_arrays_and_back
def eigh(x, /, *, lower=True, symmetrize_input=True, sort_eigenvalues=True):
    UPLO = "L" if lower else "U"

    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.eigh(x, UPLO=UPLO)


@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)


@to_ivy_arrays_and_back
def triangular_solve(a, b, left_side=False, lower=False, transpose_a=False, conjugate_a=False, unit_diagonal=False):
    """
    Solves the equation ax = b for x, where a is a triangular matrix.
    :param a: A batch of matrices with shape [..., m, m].
    :param b: A batch of matrices with shape [..., m, n] if left_side is True or shape [..., n, m] otherwise.
    :param left_side: Describes which of the two matrix equations to solve.
    :param lower: Describes which triangle of a should be used. The other triangle is ignored.
    :param transpose_a: If True, the value of a is transposed.
    :param conjugate_a: If True, the complex conjugate of a is used in the solve. Has no effect if a is real.
    :param unit_diagonal: If True, the diagonal of a is assumed to be unit (all 1s) and not accessed.
    :return: A batch of matrices the same shape and dtype as b.
    """
    return ivy.triangular_solve(a, b, left_side=left_side, lower=lower,
                                       transpose_a=transpose_a, conjugate_a=conjugate_a,
                                       unit_diagonal=unit_diagonal)