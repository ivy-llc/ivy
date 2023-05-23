# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def empty(
    *args,
    size=None,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    memory_format=None,
):
    if args and size:
        raise TypeError("empty() got multiple values for argument 'shape'")
    if size is None:
        size = args[0] if isinstance(args[0], (tuple, list)) else args
    return ivy.empty(shape=size, dtype=dtype, device=device, out=out)


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
def ones(*args, size=None, out=None, dtype=None, device=None, requires_grad=False):
    if args and size:
        raise TypeError("ones() got multiple values for argument 'shape'")
    if size is None:
        size = args[0] if isinstance(args[0], (tuple, list)) else args
    return ivy.ones(shape=size, dtype=dtype, device=device, out=out)


@to_ivy_arrays_and_back
def ones_like_v_0p3p0_to_0p3p1(input, out=None):
    return ivy.ones_like(input, out=None)


@to_ivy_arrays_and_back
def heaviside(input, values, *, out=None):
    return ivy.heaviside(input, values, out=out)


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
def zeros(*args, size=None, out=None, dtype=None, device=None, requires_grad=False):
    if args and size:
        raise TypeError("zeros() got multiple values for argument 'shape'")
    if size is None:
        size = args[0] if isinstance(args[0], (tuple, list)) else args
    return ivy.zeros(shape=size, dtype=dtype, device=device, out=out)


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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
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
        ivy.utils.assertions.check_true(
            len(args) == 1 or len(args) == 3,
            "only 1 or 3 positional arguments are supported",
        )
    return ivy.arange(start, end, step, dtype=dtype, device=device)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
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
    elif len(args) == 2:
        end = args[1]
        start = args[0]
        step = 1
    elif len(args) == 3:
        start, end, step = args
    else:
        ivy.utils.assertions.check_true(
            len(args) == 1 or len(args) == 3,
            "only 1 or 3 positional arguments are supported",
        )
    range_vec = []
    elem = start
    while 1:
        range_vec = range_vec + [elem]
        elem += step
        if start == end:
            break
        if start < end:
            if elem > end:
                break
        else:
            if elem < end:
                break
    return ivy.array(range_vec, dtype=dtype, device=device)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
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
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
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
def from_dlpack(ext_tensor):
    return ivy.from_dlpack(ext_tensor)


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
def as_strided(input, size, stride, storage_offset=None):
    ind = ivy.array([0], dtype=ivy.int64)
    for i, (size_i, stride_i) in enumerate(zip(size, stride)):
        r_size = [1] * len(stride)
        r_size[i] = -1
        ind = ind + ivy.reshape(ivy.arange(size_i), r_size) * stride_i
    if storage_offset:
        ind = ind + storage_offset
    return ivy.gather(ivy.flatten(input), ind)


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


@to_ivy_arrays_and_back
def asarray(
    obj,
    *,
    dtype=None,
    device=None,
    copy=None,
):
    return ivy.asarray(obj, copy=copy, dtype=dtype, device=device)


@to_ivy_arrays_and_back
def frombuffer(
    buffer,
    *,
    dtype,
    count=-1,
    offset=0,
    requires_grad=False,
):
    return ivy.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
