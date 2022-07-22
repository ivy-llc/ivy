# local
import ivy


def ones(size, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype:
        dtype = ivy.as_ivy_dtype(dtype)
    return ivy.ones(size, dtype=dtype, device=device, out=out)


ones.unsupported_dtypes = ("float16",)
