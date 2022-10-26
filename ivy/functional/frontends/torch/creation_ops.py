# local
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def empty(
    size,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=None,
):
    ret = ivy.empty(shape=size, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def full(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=None,
):
    ret = ivy.full(
        shape=size, fill_value=fill_value, dtype=dtype, device=device, out=out
    )
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def ones(size, *, out=None, dtype=None, device=None, requires_grad=False):
    ret = ivy.ones(shape=size, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def ones_like_v_0p3p0_to_0p3p1(input, out=None):
    return ivy.ones_like(input, out=None)


@to_ivy_arrays_and_back
def ones_like_v_0p4p0_and_above(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    ret = ivy.ones_like(input, dtype=dtype, device=device)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def zeros(size, *, out=None, dtype=None, device=None, requires_grad=False):
    ret = ivy.zeros(shape=size, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def arange(
    end,  # torch doesn't have a default for this.
    start=0,
    step=1,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    ret = ivy.arange(start, stop=end, step=step, dtype=dtype, device=device)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def range(
    end,  # torch doesn't have a default for this.
    start=0,
    step=1,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    ret = arange(start, stop=end, step=step, dtype=dtype, device=device)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def linspace(
    start,
    end,
    steps,
    *,
    out=None,
    dtype=None,
    device=None,
    layout=None,
    requires_grad=False,
):
    ret = ivy.linspace(start, end, num=steps, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def logspace(
    start,
    end,
    steps,
    *,
    base=10.0,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    ret = ivy.logspace(
        start, end, num=steps, base=base, dtype=dtype, device=device, out=out
    )
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def eye(
    n, m=None, *, out=None, dtype=None, layout=None, device=None, requires_grad=False
):
    ret = ivy.eye(n_rows=n, n_columns=m, dtype=dtype, device=device, out=out)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def empty_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    ret = ivy.empty_like(input, dtype=dtype, device=device)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def full_like(
    input,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    ret = ivy.full_like(input, fill_value=fill_value, dtype=dtype, device=device)
    if requires_grad:
        return ivy.variable(ret)
    return ret


@to_ivy_arrays_and_back
def as_tensor(
    data,
    *,
    dtype=None,
    device=None,
):
    return ivy.asarray(data, dtype=dtype, device=device)
