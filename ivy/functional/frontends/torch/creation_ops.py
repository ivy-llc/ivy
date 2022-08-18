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
    requires_grad=None
):
    ret = ivy.full(
        shape=size, fill_value=fill_value, dtype=dtype, device=device, out=out
    )
    if requires_grad:
        return ivy.variable(ret)
    return ret
