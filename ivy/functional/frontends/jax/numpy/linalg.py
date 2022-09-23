# local
import ivy


def svd(x, /, *, full_matrices=True, compute_uv=True, hermitian=False):
    return ivy.svd(
        x, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
    )
