import ivy
from ivy.functional.frontends.jax.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)


@inputs_to_ivy_arrays
def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


@inputs_to_ivy_arrays
def eigh(x, /, *, lower=True, symmetrize_input=True, sort_eigenvalues=True):
    UPLO = "L" if lower else "U"

    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.eigh(x, UPLO=UPLO)
