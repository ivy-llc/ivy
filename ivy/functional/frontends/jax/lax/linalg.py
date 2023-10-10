import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


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
@with_unsupported_dtypes({"0.4.14 and below": ("bfloat16",)}, "jax")
def qr(x, /, *, full_matrices=False):
    mode = "reduced"
    if full_matrices is True:
        mode = "complete"
    return ivy.qr(x, mode=mode)


@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)
