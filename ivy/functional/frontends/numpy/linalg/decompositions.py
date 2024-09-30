# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
def cholesky(a):
    return ivy.cholesky(a)


@to_ivy_arrays_and_back
def qr(a, mode="reduced"):
    return ivy.qr(a, mode=mode)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "1.26.3 and below": (
            "float64",
            "float32",
            "half",
            "complex32",
            "complex64",
            "complex128",
        )
    },
    "numpy",
)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    # Todo: hermitian handling
    if compute_uv:
        return ivy.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    else:
        return ivy.astype(ivy.svdvals(a), a.dtype)
