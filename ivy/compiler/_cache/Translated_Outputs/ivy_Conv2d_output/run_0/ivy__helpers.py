from itertools import repeat
import collections
import functools
import ivy
import math
import re
import warnings


def ivy__reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))


def ivy_empty_frnt(
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


def ivy_dim_frnt_(arr):
    return arr.ndim


def ivy_size_frnt_(arr, dim=None):
    shape = arr.shape
    if dim is None:
        return shape
    try:
        return shape[dim]
    except IndexError as e:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{len(shape)}, {len(shape) - 1}], but got {dim}"
        ) from e


def ivy__calculate_fan_in_and_fan_out(tensor):
    dimensions = ivy_dim_frnt_(tensor)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = ivy_size_frnt_(tensor, 1)
    num_output_fmaps = ivy_size_frnt_(tensor, 0)
    receptive_field_size = 1
    if ivy_dim_frnt_(tensor) > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def ivy__calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")
    fan_in, fan_out = ivy__calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def ivy_calculate_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def ivy_uniform__frnt_(arr, from_=0, to=1, *, generator=None):
    ret = ivy.random_uniform(
        low=from_, high=to, shape=arr.shape, dtype=arr.dtype, seed=generator
    )
    arr = ivy.inplace_update(arr, ivy.astype(ret, arr.dtype)).data
    return arr


def ivy_kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = ivy__calculate_correct_fan(tensor, mode)
    gain = ivy_calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return ivy_uniform__frnt_(tensor, -bound, bound, generator=generator)


def ivy__no_grad_uniform_(tensor, a, b, generator=None):
    return ivy_uniform__frnt_(tensor, a, b, generator=generator)


def ivy_uniform_(tensor, a=0.0, b=1.0, generator=None):
    return ivy__no_grad_uniform_(tensor, a, b, generator)


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


def ivy_parse(x):
    n = 2
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def ivy__conv_frnt(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
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


def ivy_conv2d_frnt(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    return ivy__conv_frnt(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def ivy__handle_padding_shape_frnt(padding, n, mode):
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


def ivy_pad_frnt(input, pad, mode="constant", value=0):
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
    pad = ivy__handle_padding_shape_frnt(pad, len(input.shape), mode)
    return ivy.pad(input, pad, mode=mode_dict[mode], constant_values=value)
