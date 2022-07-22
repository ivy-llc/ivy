# local
import ivy


def ones(size, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    return ivy.ones(size, dtype=dtype, device=device, out=out)


ones.unsupported_dtypes = ("float16",)
