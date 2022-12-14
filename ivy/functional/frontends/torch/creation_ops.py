# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
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
    return ret


@to_ivy_arrays_and_back
def ones(size, *, out=None, dtype=None, device=None, requires_grad=False):
    ret = ivy.ones(shape=size, dtype=dtype, device=device, out=out)
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
    return ret


@to_ivy_arrays_and_back
def zeros(size, *, out=None, dtype=None, device=None, requires_grad=False):
    ret = ivy.zeros(shape=size, dtype=dtype, device=device, out=out)
    return ret


@to_ivy_arrays_and_back
def zeros_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    ret = ivy.zeros_like(input, dtype=dtype, device=device)
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def arange(
    *args,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    if len(args) == 1:
        end = args[0]
        start = 0
        step = 1
    elif len(args) == 3:
        start, end, step = args
    else:
        ivy.assertions.check_true(
            len(args) == 1 or len(args) == 3,
            "only 1 or 3 positional arguments are supported",
        )
    return ivy.arange(start, end, step, dtype=dtype, device=device)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def range(
    *args,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    if len(args) == 1:
        end = args[0]
        start = 0
        step = 1
    elif len(args) == 3:
        start, end, step = args
    else:
        ivy.assertions.check_true(
            len(args) == 1 or len(args) == 3,
            "only 1 or 3 positional arguments are supported",
        )
    return ivy.arange(start, end, step, dtype=dtype, device=device)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
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
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
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
    return ret


@to_ivy_arrays_and_back
def eye(
    n, m=None, *, out=None, dtype=None, layout=None, device=None, requires_grad=False
):
    ret = ivy.eye(n_rows=n, n_columns=m, dtype=dtype, device=device, out=out)
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
    return ivy.full_like(input, fill_value, dtype=dtype, device=device)


@to_ivy_arrays_and_back
def as_tensor(
    data,
    *,
    dtype=None,
    device=None,
):
    return ivy.asarray(data, dtype=dtype, device=device)


@to_ivy_arrays_and_back
def from_numpy(data, /):
    return ivy.asarray(data, dtype=ivy.dtype(data))


from_numpy.supported_dtypes = ("ndarray",)


@to_ivy_arrays_and_back
def tensor(
    data,
    *,
    dtype=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    return ivy.array(data, dtype=dtype, device=device)
