import ivy

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def roots(p):
    if isinstance(p, list):
        p = ivy.asarray(p)

    ftype = ivy.dtype(p)
    if ftype in ["float64", "complex128"]:
        ftype = "complex128"
    else:
        ftype = "complex64"

    if p.size < 2:
        return ivy.array([], dtype=ivy.float64)
    A = ivy.diag(
        ivy.ones((p.size - 2), dtype=p.dtype),
        k=-1,
    )

    A[0, :] = -p[1:] / p[0]
    ret = ivy.eigvals(A)
    if p.size < 3:
        return ret.astype(p.dtype)
    else:
        return ret.astype(ftype)
