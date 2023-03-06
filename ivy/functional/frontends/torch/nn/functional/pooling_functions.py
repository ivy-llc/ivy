# global
from itertools import product
import math

# local
import ivy
from ivy import with_unsupported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
)


def _broadcast_pooling_helper(x, pool_dims: str = "2d", name: str = "padding"):
    dims = {"1d": 1, "2d": 2, "3d": 3}

    if isinstance(x, int):
        return tuple([x for _ in range(dims[pool_dims])])

    if len(x) == 1:
        return tuple([x[0] for _ in range(dims[pool_dims])])
    elif len(x) == dims[pool_dims]:
        return tuple(x)
    elif len(x) != dims[pool_dims]:
        raise ValueError(
            f"`{name}` must either be a single int, "
            f"or a tuple of {dims[pool_dims]} ints. "
        )


@to_ivy_arrays_and_back
def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # Figure out input dims N
    input_rank = input.ndim

    if input_rank == 3:
        # CHW
        data_format = "CHW"
    elif input_rank == 4:
        # NCHW
        data_format = "NCHW"

    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    stride = _broadcast_pooling_helper(stride, "2d", name="stride")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    kernel_pads = list(zip(kernel_size, padding))

    # Padding should be less than or equal to half of kernel size
    if not all([pad <= kernel / 2 for kernel, pad in kernel_pads]):
        raise ValueError(
            "pad should be smaller than or equal to half of kernel size, "
            f"but got padding={padding}, kernel_size={kernel_size}. "
        )

    # Figure out padding string
    if all([pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in kernel_pads]):
        padding_str = "SAME"
    else:
        padding_str = "VALID"

    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding_str,
        data_format=data_format,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    # ToDo: Add return_indices once superset in implemented
    dim_check = False
    if input.ndim == 3:
        input = input.expand_dims()
        dim_check = True
    if not stride:
        stride = kernel_size
    ret = ivy.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    if dim_check:
        return ret.squeeze(0)
    return ret


def kernels(ind, outd):
    def start_index(a, b, c):
        return math.floor((float(a) * float(c)) / b)

    def end_index(a, b, c):
        return math.ceil((float(a + 1) * float(c)) / b)

    results = []
    for ow in range(outd):
        start = start_index(ow, outd, ind)
        end = end_index(ow, outd, ind)
        sz = end - start
        results.append((start, sz))
    return results


def kernel_indexes(ind, out):
    startsLengths = kernels(ind, out)
    return [list(range(start, start + length)) for (start, length) in startsLengths]


# Reference: https://stackoverflow.com/a/63603993
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool1d(input, output_size):
    squeeze = False
    if len(input.shape) == 2:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 3:
        raise ivy.utils.exceptions.IvyException(
            f"Got {len(input.shape)}D input, but only 2D and 3D inputs are supported.",
        )
    input_size = input.shape[-1]
    if input_size % output_size == 0:
        stride = input_size // output_size
        kernel_size = input_size - (output_size - 1) * stride
        pooled_output = ivy.avg_pool1d(
            input, kernel_size, stride, "VALID", data_format="NCW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output
    else:
        kernels = kernel_indexes(input_size, output_size)
        pooled_output = ivy.stack(
            [sum([input[:, :, x] for x in xs]) / len(xs) for xs in kernels], axis=-1
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):

    device = input.device
    shape = input.shape
    squeeze = False

    if len(input.shape) == 3:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 4:
        raise ivy.utils.exceptions.IvyException(
            f"Got {len(shape)}D input, but only 3D and 4D inputs are supported.",
        )
    for d in input.shape[-2:]:
        if d == 0:
            raise ivy.utils.exceptions.IvyException(
                "Expected input to have non-zero size for non-batch dimensions, but"
                f" input has shape {tuple(shape)}."
            )

    if all(i_s % o_s == 0 for i_s, o_s in zip(shape[-2:], output_size)):
        stride = tuple(i_s // o_s for i_s, o_s in zip(shape[-2:], output_size))
        kernel_size = tuple(
            i_s - (o_s - 1) * st
            for i_s, o_s, st in zip(shape[-2:], output_size, stride)
        )
        pooled_output = ivy.avg_pool2d(
            input, kernel_size, stride, "VALID", data_format="NCHW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output

    def start_index(a, b, c):
        return ivy.trunc_divide(a * c, b).astype(ivy.int64)

    def end_index(a, b, c):
        return ivy.trunc_divide((a + 1) * c + b - 1, b).astype(ivy.int64)

    def compute_idx(in_size, out_size):
        orange = ivy.arange(out_size, device=device, dtype=ivy.int64)
        i0 = start_index(orange, out_size, in_size)
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1
        range_max = ivy.arange(maxlength, device=device, dtype=ivy.int64)
        idx = ivy.expand_dims(i0, axis=-1) + range_max
        if adaptive:
            maxval = ivy.full_like(idx, fill_value=in_size - 1)
            idx = ivy.minimum(idx, maxval)
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    def _expand_to_dim(x, dim):
        for _ in range(dim - len(x.shape)):
            x = ivy.expand_dims(x, axis=-1)
        return x

    idxh, length_h, range_max_h, adaptive_h = compute_idx(shape[-2], output_size[-2])
    idxw, length_w, range_max_w, adaptive_w = compute_idx(shape[-1], output_size[-1])

    # to numpy and back in order to bypass a slicing error in tensorflow
    vals = ivy.array(input.to_numpy()[..., _expand_to_dim(idxh, 4), idxw])

    if not adaptive_h and not adaptive_w:
        return ivy.mean(vals, axis=(-3, -1))

    def maybe_mask(vals, length, range_max, dim):
        if isinstance(length, int):
            return vals, length
        else:
            assert dim < 0
            mask = ivy.greater_equal(range_max, ivy.expand_dims(length, axis=-1))
            if dim == -2:
                mask = _expand_to_dim(mask, 4)
            vals = ivy.where(mask, 0.0, vals)
            length = _expand_to_dim(length, -dim)
            return vals, length

    vals, length_h = maybe_mask(vals, length_h, range_max_h, dim=-2)
    vals, length_w = maybe_mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i, j in product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    pooled_output = ret / (length_h * length_w)

    if squeeze:
        return ivy.squeeze(pooled_output, axis=0)
    return pooled_output
