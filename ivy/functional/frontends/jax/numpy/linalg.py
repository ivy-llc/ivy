# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import inputs_to_ivy_arrays


@inputs_to_ivy_arrays
def inv(a):
    return ivy.inv(a)


@inputs_to_ivy_arrays
def det(a):
    return ivy.det(a)


@inputs_to_ivy_arrays
def eigh(a, UPLO="L", symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        a = symmetrize(a)

    return ivy.eigh(a, UPLO=UPLO)


# eigvalsh
def eigvalsh(a, UPLO="L"):
    return ivy.eigvalsh(a, UPLO=UPLO)


def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


def eigvals(a):
    return ivy.eigh(a)
