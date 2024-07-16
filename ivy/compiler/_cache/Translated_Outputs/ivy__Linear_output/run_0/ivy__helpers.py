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


def ivy_linear(input, weight, bias=None):
    return ivy.linear(input, weight, bias=bias)


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
