# global
from functools import reduce

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool1d(input, output_size):
    return ivy.adaptive_avg_pool1d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):
    return ivy.adaptive_avg_pool2d(input, output_size, data_format="NCHW")


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_max_pool2d(
    input,
    output_size,
    return_indices=False,
):
    # ToDo: Add return_indices once superset is implemented
    return ivy.adaptive_max_pool2d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "int8",
            "int16",
            "bool",
            "uint8",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def adaptive_max_pool3d(
    input,
    output_size,
    return_indices=False,
):
    return ivy.adaptive_max_pool3d(input, output_size)


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    "torch",
)
@to_ivy_arrays_and_back
def avg_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool1d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCW",
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    "torch",
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
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "bfloat16")},
    "torch",
)
@to_ivy_arrays_and_back
def avg_pool3d(
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
    return ivy.avg_pool3d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCDHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size

    out = ivy.avg_pool1d(
        ivy.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    p = 1.0 / norm_type if norm_type != 0 else 1.0
    return ivy.pow(ivy.multiply(out, kernel_mul), p)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCHW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    out = ivy.avg_pool2d(
        ivy.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size
    p = ivy.divide(1.0, norm_type) if norm_type != 0 else 1.0
    return ivy.pow(ivy.multiply(out, kernel_mul), p).astype(input.dtype)


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    if input.ndim == 2:
        without_batch_dim = True
        input = ivy.expand_dims(input, axis=0)
    else:
        without_batch_dim = False

    ret = ivy.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    if without_batch_dim:
        ret = ret[0]
    return ret


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
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
    if not stride:
        stride = kernel_size
    if input.ndim == 3:
        without_batch_dim = True
        input = ivy.expand_dims(input, axis=0)
    else:
        without_batch_dim = False

    output = ivy.max_pool2d(
        input,
        kernel_size,
        stride,
        ([(pad, pad) for pad in padding] if not isinstance(padding, int) else padding),
        data_format="NCHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    if return_indices:
        if isinstance(stride, (list, tuple)) and len(stride) == 1:
            stride = stride[0]

        DIMS = 2
        x_shape = list(input.shape[2:])
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        new_kernel = [
            kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)
            for i in range(DIMS)
        ]

        if isinstance(padding, int):
            padding = [(padding,) * 2] * DIMS
        elif isinstance(padding, (list, tuple)) and len(padding) == DIMS:
            padding = [(padding[i],) * 2 for i in range(DIMS)]

        if isinstance(stride, int):
            stride = (stride,) * DIMS

        if ceil_mode:
            for i in range(DIMS):
                padding[i] = ivy.functional.ivy.experimental.layers._padding_ceil_mode(
                    x_shape[i], new_kernel[i], padding[i], stride[i]
                )
        # torch pad takes width padding first, then height padding
        padding = (padding[1], padding[0])
        pad_list = list(ivy.flatten(padding))

        in_shape = input.shape
        H = in_shape[-2]
        W = in_shape[-1]
        n_indices = H * W

        # calculate the indices within the input tensor
        # for each position in the sliding window
        input_indices = torch_frontend.arange(0, n_indices, dtype=torch_frontend.int64)
        input_indices = input_indices.reshape((1, 1, H, W))

        # find the indices of the max value for each position of the sliding window
        input = torch_frontend.nn.functional.pad(
            input,
            pad_list,
            value=float("-inf"),
        )

        input_indices = torch_frontend.nn.functional.pad(
            input_indices,
            pad_list,
            value=0,
        )

        unfolded_indices = torch_frontend.nn.functional.unfold(
            input_indices,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            stride=stride,
        ).permute((0, 2, 1))[0]

        unfolded_values = torch_frontend.nn.functional.unfold(
            input, kernel_size=kernel_size, padding=0, dilation=dilation, stride=stride
        )
        unfolded_values_shape = unfolded_values.shape
        unfolded_indices = unfolded_indices.repeat(
            unfolded_values_shape[0], unfolded_values_shape[1], 1, 1
        )
        unfolded_values = unfolded_values.reshape(
            input.shape[0],
            input.shape[1],
            unfolded_values.shape[1] // input.shape[1],
            unfolded_values.shape[2],
        )
        indices = torch_frontend.argmax(unfolded_values, dim=2)

        # gather the indices within the input tensor of the max values
        indices = torch_frontend.gather(
            unfolded_indices, -1, torch_frontend.unsqueeze(indices, -1)
        )
        indices = indices.reshape(output.shape)

    if without_batch_dim:
        output = output[0]
        if return_indices:
            indices = indices[0]

    if return_indices:
        return output, indices
    return output


@with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    if input.ndim == 4:
        without_batch_dim = True
        input = ivy.expand_dims(input, axis=0)
    else:
        without_batch_dim = False

    ret = ivy.max_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCDHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    if without_batch_dim:
        ret = ret[0]
    return ret
