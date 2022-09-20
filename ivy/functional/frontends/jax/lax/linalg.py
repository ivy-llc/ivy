import ivy


def _symmetrize(x):
    # TODO : Take Hermitian transpose after complex numbers added
    return (x + ivy.swapaxes(x, -1, -2)) / 2


def cholesky(x, /, *, symmetrize_input=True):
    if symmetrize_input:
        x = _symmetrize(x)

    return ivy.cholesky(x)


def eigh(x, /, *, lower=True, symmetrize_input=True, sort_eigenvalues=True):
    upper = not lower
    
    if symmetrize_input:
        x = _symmetrize(x)

    if sort_eigenvalues:
        ret = ivy.eigh(x, upper)
        return ret._replace(eigenvalues = ivy.sort(ret.eigenvalues)

    return ivy.eigh(x, upper)
