# global
import ivy


def cholesky(a):
    return ivy.cholesky(a)


cholesky.unsupported_dtypes = ("float16",)


def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


qr.unsupported_dtypes = ("float16",)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    # Todo: conpute_uv and hermitian handling
    return ivy.svd(a, full_matrices=full_matrices)


svd.unsupported_dtypes = ("float16",)
