# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)


@to_ivy_arrays_and_back
def det(a):
    return ivy.det(a)


@to_ivy_arrays_and_back
def eigh(a, UPLO="L", symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        a = symmetrize(a)

    return ivy.eigh(a, UPLO=UPLO)


@to_ivy_arrays_and_back
def eigvalsh(a, UPLO="L"):
    return ivy.eigvalsh(a, UPLO=UPLO)


@to_ivy_arrays_and_back
def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


@to_ivy_arrays_and_back
def eigvals(a):
    return ivy.eigh(a)


@to_ivy_arrays_and_back
def cholesky(a):
    return ivy.cholesky(a)


@to_ivy_arrays_and_back
def slogdet(a, method=None):
    return ivy.slogdet(a)


@to_ivy_arrays_and_back
def matrix_rank(M):
    return ivy.matrix_rank(M)
