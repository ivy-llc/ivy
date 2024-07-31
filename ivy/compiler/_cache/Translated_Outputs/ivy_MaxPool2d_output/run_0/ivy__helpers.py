import functools
import ivy
import math
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


def ivy_dim_frnt_(arr):
    return arr.ndim


def ivy_ndim_frnt_(arr):
    return ivy_dim_frnt_(arr)


def ivy_arange_frnt(
    start=0,
    end=None,
    step=1,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    return ivy.arange(start, end, step, dtype=dtype, device=device, out=out)


def ivy_reshape_frnt(input, shape):
    return ivy.reshape(input, shape)


def ivy_reshape_frnt_(arr, *args, shape=None):
    if args and shape:
        raise TypeError("reshape() got multiple values for argument 'shape'")
    if shape is not None:
        return ivy_reshape_frnt(arr, shape)
    if args:
        if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
            shape = args[0]
            return ivy_reshape_frnt(arr, shape)
        else:
            return ivy_reshape_frnt(arr, args)
    else:
        raise ValueError("reshape() got no values for argument 'shape'")


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


def ivy_permute_frnt(input, dims):
    return ivy.permute_dims(input, axes=dims, copy=False)


def ivy_permute_frnt_(arr, *args, dims=None):
    if args and dims:
        raise TypeError("permute() got multiple values for argument 'dims'")
    if dims is not None:
        return ivy_permute_frnt(arr, dims)
    if args:
        if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
            dims = args[0]
            return ivy_permute_frnt(arr, dims)
        else:
            return ivy_permute_frnt(arr, args)
    else:
        raise ValueError("permute() got no values for argument 'dims'")


def ivy_unfold_frnt(input, kernel_size, dilation=1, padding=0, stride=1):
    if ivy_ndim_frnt_(input) != 4:
        raise ivy.utils.exceptions.IvyException("only batched 4D inputs are supported")
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_shape = [
        (
            (
                input.shape[i + 2]
                + 2 * padding[i]
                - dilation[i] * (kernel_size[i] - 1)
                - 1
            )
            // stride[i]
            + 1
        )
        for i in range(2)
    ]
    ret = ivy.zeros((*input.shape[0:2], *kernel_size, *output_shape), dtype=input.dtype)
    input_padded = ivy.zero_pad(
        input, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2)
    )
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            ret[:, :, :, :, i, j] = input_padded[
                :,
                :,
                i_in : i_in + kernel_size[0] * dilation[0] : dilation[0],
                j_in : j_in + kernel_size[1] * dilation[1] : dilation[1],
            ]
    return ivy.reshape(
        ret, (input.shape[0], input.shape[1] * math.prod(kernel_size), -1)
    )


def ivy_tile_frnt(input, dims):
    try:
        tup = tuple(dims)
    except TypeError:
        tup = (dims,)
    d = len(tup)
    res = 0
    if len(input.shape) > len([dims]) - 1:
        res = input
    if d < ivy_ndim_frnt_(input):
        tup = (1,) * (ivy_ndim_frnt_(input) - d) + tup
        res = ivy.tile(input, tup)
    else:
        res = ivy.tile(input, repeats=dims, out=None)
    return res


def ivy_repeat_frnt_(arr, *args, repeats=None):
    if args and repeats:
        raise ivy.utils.exceptions.IvyException(
            "repeat() got multiple values for argument 'repeats'"
        )
    if args:
        if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
            repeats = args[0]
        else:
            repeats = args
    elif not isinstance(repeats, (tuple, list)):
        raise ivy.utils.exceptions.IvyException(
            "repeat(): argument 'repeats' must be tuple of ints"
        )
    return ivy_tile_frnt(arr, repeats)


def ivy_argmax_frnt(input, dim=None, keepdim=False):
    return ivy.argmax(input, axis=dim, keepdims=keepdim)


def ivy_gather_frnt(input, dim, index, *, sparse_grad=False, out=None):
    if sparse_grad:
        raise ivy.utils.exceptions.IvyException(
            "Gather does not yet support the sparse grad functionality"
        )
    dim = dim % len(input.shape)
    all_indices = ivy.argwhere(ivy.full(index.shape, True))
    gather_locations = ivy.reshape(
        index, [ivy.prod(ivy.array(index.shape), dtype=ivy.int64)]
    )
    gather_indices = []
    for axis in range(len(index.shape)):
        if axis == dim:
            gather_indices.append(ivy.array(gather_locations, dtype=index.dtype))
        else:
            gather_indices.append(ivy.array(all_indices[:, axis], dtype=index.dtype))
    gather_indices = ivy.stack(gather_indices, axis=-1)
    gathered = ivy.gather_nd(input, gather_indices)
    reshaped = ivy.reshape(gathered, index.shape)
    return reshaped


def ivy_unsqueeze_frnt(input, dim=0):
    return ivy.expand_dims(input, axis=dim)


def ivy_max_pool2d_frnt(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if not stride:
        stride = kernel_size
    if ivy_ndim_frnt_(input) == 3:
        without_batch_dim = True
        input = ivy.expand_dims(input, axis=0)
    else:
        without_batch_dim = False
    output = ivy.max_pool2d(
        input,
        kernel_size,
        stride,
        [(pad, pad) for pad in padding] if not isinstance(padding, int) else padding,
        data_format="NHWC",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    if return_indices:
        if isinstance(stride, (list, tuple)) and len(stride) == 1:
            stride = stride[0]
        DIMS = 2
        x_shape = list(input.shape[2:])
        new_kernel = [
            (kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1))
            for i in range(DIMS)
        ]
        if isinstance(padding, int):
            padding = [(padding,) * 2] * DIMS
        elif isinstance(padding, (list, tuple)) and len(padding) == DIMS:
            padding = [((padding[i],) * 2) for i in range(DIMS)]
        if isinstance(stride, int):
            stride = (stride,) * DIMS
        if ceil_mode:
            for i in range(DIMS):
                padding[i] = ivy.functional.ivy.experimental.layers._padding_ceil_mode(
                    x_shape[i], new_kernel[i], padding[i], stride[i]
                )
        padding = padding[1], padding[0]
        pad_list = list(ivy.flatten(padding))
        in_shape = input.shape
        H = in_shape[-2]
        W = in_shape[-1]
        n_indices = H * W
        input_indices = ivy_arange_frnt(0, n_indices, dtype=ivy.int64)
        input_indices = ivy_reshape_frnt_(input_indices, (1, 1, H, W))
        input = ivy_pad_frnt(input, pad_list, value=float("-inf"))
        input_indices = ivy_pad_frnt(input_indices, pad_list, value=0)
        unfolded_indices = ivy_permute_frnt_(
            ivy_unfold_frnt(
                input_indices,
                kernel_size=kernel_size,
                padding=0,
                dilation=dilation,
                stride=stride,
            ),
            (0, 2, 1),
        )[0]
        unfolded_values = ivy_unfold_frnt(
            input, kernel_size=kernel_size, padding=0, dilation=dilation, stride=stride
        )
        unfolded_values_shape = unfolded_values.shape
        unfolded_indices = ivy_repeat_frnt_(
            unfolded_indices, unfolded_values_shape[0], unfolded_values_shape[1], 1, 1
        )
        unfolded_values = ivy_reshape_frnt_(
            unfolded_values,
            input.shape[0],
            input.shape[1],
            unfolded_values.shape[1] // input.shape[1],
            unfolded_values.shape[2],
        )
        indices = ivy_argmax_frnt(unfolded_values, dim=2)
        indices = ivy_gather_frnt(unfolded_indices, -1, ivy_unsqueeze_frnt(indices, -1))
        indices = ivy_reshape_frnt_(indices, output.shape)
    if without_batch_dim:
        output = output[0]
        if return_indices:
            indices = indices[0]
    if return_indices:
        return output, indices
    return output
