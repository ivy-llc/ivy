import ivy


# inv
def inv(a):
    return ivy.inv(a)


def det(a):
    return ivy.det(a)


def eigh(a, UPLO="L", symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        a = symmetrize(a)

    return ivy.eigh(a, UPLO=UPLO)


def svd(x, /, *, full_matrices=True, compute_uv=True, hermitian=False):
    return ivy.svd(
        x, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
    )
