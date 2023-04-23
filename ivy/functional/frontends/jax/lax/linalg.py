import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def svd(x, /, *, full_matrices=True, compute_uv=True):
    if not compute_uv:
        return ivy.svdvals(x)
    return ivy.svd(x, full_matrices=full_matrices)


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
def qdwh(x, eps=1e-6, max_iterations=100):
    m, n = x.shape
    X = x.copy
    for i in range(max_iterations):
        Q, R = ivy.qr(X)
        Y = ivy.multi_dot(R, Q)
        Z = 0.5 * (Y + ivy.linalg.inv(Y.T))
        weight = ivy.where(ivy.abs(Y - Z) < eps, 1.0, ivy.abs(Y - Z) / ivy.abs(Y))
        X = ivy.multi_dot(ivy.multi_dot(Z, (2 * ivy.eye(n) - ivy.multi_dot(Z, X))), weight)
        normal = ivy.linalg.matrix_norm(ivy.multi_dot(X.T, X) - ivy.eye(n))
        if normal < eps: break

    U, _, V = ivy.linalg.svd(X)

    return ivy.qdwh(x)

