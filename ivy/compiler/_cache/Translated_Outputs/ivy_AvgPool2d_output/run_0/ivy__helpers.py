import functools
import ivy
import re


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
            pattern = "_bknd_|_bknd|_frnt_|_frnt"
            fn_name = extract_function_name(re.sub(pattern, "", fn.__name__))
            new_fn = getattr(args[0], fn_name)
            return new_fn(*args[1:], **kwargs)

    return wrapper


@ivy_handle_methods
def ivy_split_frnt(tensor, split_size_or_sections, dim=0):
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
def ivy_split_frnt_(arr, split_size, dim=0):
    return ivy_split_frnt(arr, split_size, dim)


@ivy_handle_methods
def ivy_add_frnt(input, other, *, alpha=1, out=None):
    return ivy.add(input, other, alpha=alpha, out=out)


@ivy_handle_methods
def ivy_add_frnt_(arr, other, *, alpha=1):
    return ivy_add_frnt(arr, other, alpha=alpha)


def ivy_avg_pool2d_frnt(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NHWC",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
