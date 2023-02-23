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
    # Todo: conpute_uv and hermitian handling
    return ivy.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
