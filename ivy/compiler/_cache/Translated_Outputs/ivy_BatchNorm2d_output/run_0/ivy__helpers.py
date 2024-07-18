import functools
import ivy
import re


def ivy_empty(
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
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape))
            else args
        )
    if isinstance(size, (tuple, list)):
        size = tuple(s.to_scalar() if ivy.is_array(s) else s for s in size)
    return ivy.empty(shape=size, dtype=dtype, device=device, out=out)


def ivy_zeros(*args, size=None, out=None, dtype=None, device=None, requires_grad=False):
    if args and size:
        raise TypeError("zeros() got multiple values for argument 'shape'")
    if size is None:
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape))
            else args
        )
    return ivy.zeros(shape=size, dtype=dtype, device=device, out=out)


def ivy_ones(*args, size=None, out=None, dtype=None, device=None, requires_grad=False):
    if args and size:
        raise TypeError("ones() got multiple values for argument 'shape'")
    if size is None:
        size = (
            args[0]
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape))
            else args
        )
    return ivy.ones(shape=size, dtype=dtype, device=device, out=out)


def ivy_tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return ivy.array(data, dtype=dtype, device=device)


def ivy_zeros_like(
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


def ivy_zero_(arr):
    ret = ivy_zeros_like(arr)
    arr = ivy.inplace_update(arr, ret).data
    return arr


def ivy_full_like(
    input,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=None,
):
    fill_value = ivy.to_scalar(fill_value)
    return ivy.full_like(input, fill_value, dtype=dtype, device=device)


def ivy_fill_(arr, value):
    ret = ivy_full_like(arr, value, dtype=arr.dtype, device=arr.device)
    arr = ivy.inplace_update(arr, ret).data
    return arr


def ivy__no_grad_fill_(tensor, val):
    return ivy_fill_(tensor, val)


def ivy_ones_(tensor):
    return ivy__no_grad_fill_(tensor, 1.0)


def ivy__no_grad_zero_(tensor):
    return ivy_zero_(tensor)


def ivy_zeros_(tensor):
    return ivy__no_grad_zero_(tensor)


def ivy_device(dev):
    return ivy.default_device(dev)


def ivy_handle_methods(fn):
    def extract_function_name(s):
        match = re.search("_(.+?)(?:_\\d+)?$", s)
        if match:
            return match.group(1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if ivy.is_array(args[0]):
            return fn(*args, **kwargs)
        else:
            fn_name = extract_function_name(fn.__name__)
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


@ivy_handle_methods
def ivy_split_1(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        split_size_or_sections = [split_size] * (tensor.shape[dim] // split_size)
        if tensor.shape[dim] % split_size:
            split_size_or_sections.append(tensor.shape[dim] % split_size)
    return tuple(
        ivy.split(
            tensor,
            num_or_size_splits=split_size_or_sections,
            axis=dim,
            with_remainder=True,
        )
    )


@ivy_handle_methods
def ivy_split(arr, split_size, dim=0):
    return ivy_split_1(arr, split_size, dim)


@ivy_handle_methods
def ivy_add_1(input, other, *, alpha=1, out=None):
    return ivy.add(input, other, alpha=alpha, out=out)


@ivy_handle_methods
def ivy_add(arr, other, *, alpha=1):
    return ivy_add_1(arr, other, alpha=alpha)


def ivy_add_(arr, other, *, alpha=1):
    arr = ivy_add(arr, other, alpha=alpha)
    return arr


def ivy_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    normalized, mean, var = ivy.batch_norm(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=training,
        eps=eps,
        momentum=momentum,
        data_format="NSC",
    )
    return normalized, mean, var


def ivy_dim(arr):
    return arr.ndim
