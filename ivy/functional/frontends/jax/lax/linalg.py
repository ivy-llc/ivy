import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from jax._src import core


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
def qdwh(
    x, /, *, is_hermitian=False, max_iterations=None, eps=None, dynamic_shape=None
):
    is_hermitian = core.concrete_or_error(
        bool,
        is_hermitian,
        "The `is_hermitian` argument must be statically "
        "specified to use `qdwh` within JAX transformations.",
    )

    if max_iterations is None:
        max_iterations = 10

    M, N = x.shape

    if M < N:
        raise ValueError("The input matrix of shape M x N must have M >= N.")

    if dynamic_shape is not None:
        m, n = dynamic_shape
        x = ivy._mask(x, (m, n))

    else:
        m, n = M, N

    with ivy.jax.default_matmul_precision("float32"):
        u, h, num_iters, is_converged = ivy._qdwh(
            x, m, n, is_hermitian, max_iterations, eps
        )

    return ivy.qdwh(u, h, num_iters, is_converged)


@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)
