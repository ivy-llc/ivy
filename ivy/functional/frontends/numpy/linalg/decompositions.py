# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def cholesky(a):
    return ivy.cholesky(a)


@to_ivy_arrays_and_back
def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


@to_ivy_arrays_and_back
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    if hermitian:
        u, s = ivy.eig(a)
        v = ivy.conj(u)
    else:
        u, s, v = ivy.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    if not compute_uv:
        return s

    if not full_matrices:
        u = u[:, : s.shape[-1]]
        v = v[:, : s.shape[-1]]

    return u, s, v
