import ivy


def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svd(x, full_matrices=full_matrices)[1]
    return ivy.svd(x, full_matrices=full_matrices)
