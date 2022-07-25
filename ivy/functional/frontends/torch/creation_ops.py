# local
import ivy


def full(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    if dtype:
        dtype = ivy.as_ivy_dtype(dtype)
    ret = ivy.full(size, fill_value, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    else:
        return ret
