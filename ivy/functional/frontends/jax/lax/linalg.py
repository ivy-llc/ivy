import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes
import numpy as np_frontend


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
def qdwh(x, *, is_hermitian=False, max_iterations=None, eps=None, dynamic_shape=None):
    if dynamic_shape:
        m, n = dynamic_shape
        x_pad = np_frontend.zeros((m, n), dtype=x.dtype)
        x_pad[: x.shape[0], : x.shape[1]] = x
        x = x_pad

    # Compute the SVD of x
    u, s, vh = np_frontend.linalg.svd(x, full_matrices=False)
    v = vh.T.conj()

    # Compute the weighted average of u and v
    alpha = 0.5
    u_avg = alpha * u + (1 - alpha) * v

    # Compute the diagonal matrix h = u_avg^H * x
    h = u_avg.conj().T @ x

    # Apply the Halley iteration
    num_iters = 0
    while True:
        h_prev = h.copy()
        h2 = h @ h_prev
        h3 = h2 @ h_prev
        g = 1.5 * h_prev - 0.5 * h_prev @ h2
        delta_h = np_frontend.linalg.solve(2 * g - h3, h2 - 2 * g @ h_prev + g @ h3)
        h += delta_h
        num_iters += 1
        x = np_frontend.linalg.norm(delta_h)
        y = np_frontend.linalg.norm(h) * (4 * eps) ** (1 / 3)
        if eps:
            if x < y:
                is_converged = True
                break

        if max_iterations and num_iters >= max_iterations:
            is_converged = False
            break

    # Compute the polar decomposition
    h_sqrt = np_frontend.sqrt(h.conj().T @ h)
    u = u_avg @ np_frontend.linalg.inv(h_sqrt)

    return u, h, num_iters, is_converged


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
