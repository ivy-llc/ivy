# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
def diagonal(input, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


@to_ivy_arrays_and_back
def inv(input, *, out=None):
    return ivy.inv(input, out=out)


@to_ivy_arrays_and_back
def pinv(input, *, atol=None, rtol=None, hermitian=False, out=None):
    if atol is None:
        return ivy.pinv(input, rtol=rtol, out=out)
    else:
        sigma = ivy.svdvals(input)[0]
        if rtol is None:
            rtol = atol / sigma
        else:
            if atol > rtol * sigma:
                rtol = atol / sigma

    return ivy.pinv(input, rtol=rtol, out=out)


@to_ivy_arrays_and_back
def det(input, *, out=None):
    return ivy.det(input, out=out)


@to_ivy_arrays_and_back
def eigvalsh(input, UPLO="L", *, out=None):
    return ivy.eigvalsh(input, UPLO=UPLO, out=out)


@to_ivy_arrays_and_back
def slogdet(input, *, out=None):
    return ivy.slogdet(input, out=out)


@to_ivy_arrays_and_back
def matrix_power(input, n, *, out=None):
    return ivy.matrix_power(input, n, out=out)


@with_supported_dtypes(
    {"1.11.0 and below": ("float32", "float64", "complex64", "complex128")}, "torch"
)
@to_ivy_arrays_and_back
def matrix_norm(input, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None, out=None):
    if "complex" in ivy.as_ivy_dtype(input.dtype):
        input = ivy.abs(input)
    if dtype:
        input = ivy.astype(input, dtype)
    return ivy.matrix_norm(input, ord=ord, axis=dim, keepdims=keepdim, out=out)


@to_ivy_arrays_and_back
def matrix_rank(input, *, atol=None, rtol=None, hermitian=False, out=None):
    return ivy.astype(ivy.matrix_rank(input, atol=atol, rtol=rtol, out=out), ivy.int64)

