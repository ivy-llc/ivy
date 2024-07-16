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


def ivy__conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    dims = len(input.shape) - 2
    if isinstance(padding, str):
        padding = padding.upper()
    elif isinstance(padding, int):
        padding = [*[(padding, padding) for _ in range(dims)]]
    else:
        padding = [*[(p, p) for p in padding]]
    ret = ivy.conv_general_dilated(
        input,
        weight,
        stride,
        padding,
        dims=dims,
        data_format="channel_last",
        filter_format="channel_last",
        dilations=dilation,
        feature_group_count=groups,
        bias=bias,
    )
    return ret


def ivy_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return ivy__conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def ivy__handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    if mode == "circular":
        padding = padding + ((0, 0),) * (n - len(padding))
    else:
        padding = ((0, 0),) * (n - len(padding)) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


def ivy_pad(input, pad, mode="constant", value=0):
    if any([(pad_value < 0) for pad_value in pad]):
        pad = list(pad)
        slices = []
        for n in reversed(range(len(pad) // 2)):
            i = n * 2
            j = i + 1
            start = None
            stop = None
            if pad[i] < 0:
                start = -pad[i]
                pad[i] = 0
            if pad[j] < 0:
                stop = pad[j]
                pad[j] = 0
            slices.append(slice(start, stop))
        ndim = len(input.shape)
        while len(slices) < ndim:
            slices.insert(0, slice(None))
        input = input[tuple(slices)]
    value = 0 if value is None else value
    mode_dict = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if mode not in mode_dict:
        raise ValueError(f"Unsupported padding mode: {mode}")
    pad = ivy__handle_padding_shape(pad, len(input.shape), mode)
    return ivy.pad(input, pad, mode=mode_dict[mode], constant_values=value)
