# global
from typing import Optional, Union, Tuple, List, Literal, Sequence, Callable
import torch
import math

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from . import backend_version
from ivy.functional.ivy.layers import (
    _handle_padding,
    _get_num_padded_values,
    _validate_max_pool_params,
    _depth_max_pooling_helper,
)
from ivy.functional.ivy.experimental.layers import (
    _padding_ceil_mode,
    _broadcast_pooling_helper,
)


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_first"):
    # Determine depth pooling
    kernel, strides, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        x = torch.permute(x, (0, 2, 1, *range(3, dims + 2)))
    return x, kernel, strides, depth_pooling


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def max_pool1d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dims = 1
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    if data_format == "NWC":
        x = x.permute(0, 2, 1)
        kernel = [kernel[i] for i in [0, 2, 1]] if len(kernel) == (dims + 2) else kernel
        strides = (
            [strides[i] for i in [0, 2, 1]] if len(strides) == (dims + 2) else strides
        )

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_first"
    )

    if isinstance(padding, str):
        x_shape = list(x.shape[2:])
        new_kernel = [dilation[0] * (kernel[0] - 1) + 1]
        pad_w = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
        pad_list = [pad_w // 2, pad_w - pad_w // 2]
    else:
        if any(item != 0 for sublist in padding for item in sublist) and depth_pooling:
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        pad_list = [item for sublist in padding[::-1] for item in sublist]

    if all(pad_list[i] == pad_list[i + 1] for i in range(0, 2 * dims, 2)) and all(
        pad <= kernel_size / 2 for pad, kernel_size in zip(pad_list[::-2], kernel)
    ):
        res = torch.nn.functional.max_pool1d(
            x, kernel, strides, pad_list[::-2], dilation, ceil_mode
        )
    else:
        x = torch.nn.functional.pad(
            x,
            pad_list,
            value=float("-inf"),
        )
        res = torch.nn.functional.max_pool1d(x, kernel, strides, 0, dilation, ceil_mode)

    if depth_pooling:
        res = torch.permute(res, (0, 2, 1))
    if data_format == "NWC":
        return res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def max_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dims = 2
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
        kernel = (
            [kernel[i] for i in [0, 3, 1, 2]] if len(kernel) == (dims + 2) else kernel
        )
        strides = (
            [strides[i] for i in [0, 3, 1, 2]]
            if len(strides) == (dims + 2)
            else strides
        )

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_first"
    )

    if isinstance(padding, str):
        x_shape = list(x.shape[2:])
        new_kernel = [kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(2)]
        pad_h = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
        pad_w = _handle_padding(x_shape[1], strides[1], new_kernel[1], padding)
        pad_list = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    else:
        if any(item != 0 for sublist in padding for item in sublist) and depth_pooling:
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        pad_list = [item for sublist in padding[::-1] for item in sublist]

    if all(pad_list[i] == pad_list[i + 1] for i in range(0, 2 * dims, 2)) and all(
        pad <= kernel_size / 2 for pad, kernel_size in zip(pad_list[::-2], kernel)
    ):
        res = torch.nn.functional.max_pool2d(
            x, kernel, strides, pad_list[::-2], dilation, ceil_mode
        )
    else:
        x = torch.nn.functional.pad(
            x,
            pad_list,
            value=float("-inf"),
        )
        res = torch.nn.functional.max_pool2d(x, kernel, strides, 0, dilation, ceil_mode)

    if depth_pooling:
        res = torch.permute(res, (0, 2, 1, 3))
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def max_pool3d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dims = 3
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
        kernel = (
            [kernel[i] for i in [0, 4, 1, 2, 3]]
            if len(kernel) == (dims + 2)
            else kernel
        )
        strides = (
            [strides[i] for i in [0, 4, 1, 2, 3]]
            if len(strides) == (dims + 2)
            else strides
        )

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format="channel_first"
    )

    if isinstance(padding, str):
        x_shape = list(x.shape[2:])
        new_kernel = [kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(3)]
        pad_d = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
        pad_h = _handle_padding(x_shape[1], strides[1], new_kernel[1], padding)
        pad_w = _handle_padding(x_shape[2], strides[2], new_kernel[2], padding)
        pad_list = [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        ]
    else:
        if any(item != 0 for sublist in padding for item in sublist) and depth_pooling:
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        pad_list = [item for sublist in padding[::-1] for item in sublist]

    if all(pad_list[i] == pad_list[i + 1] for i in range(0, 2 * dims, 2)) and all(
        pad <= kernel_size / 2 for pad, kernel_size in zip(pad_list[::-2], kernel)
    ):
        res = torch.nn.functional.max_pool3d(
            x, kernel, strides, pad_list[::-2], dilation, ceil_mode
        )
    else:
        x = torch.nn.functional.pad(
            x,
            pad_list,
            value=float("-inf"),
        )
        res = torch.nn.functional.max_pool3d(x, kernel, strides, 0, dilation, ceil_mode)

    if depth_pooling:
        res = torch.permute(res, (0, 2, 1, 3, 4))
    if data_format == "NDHWC":
        return res.permute(0, 2, 3, 4, 1)
    return res


def _add_ceil_pad_to_pad_list(num_pad, k, c):
    return num_pad + (num_pad - ((k * num_pad) / (k - c)))


def _get_specific_pad(x_shape, kernel, strides, padding, dims):
    if isinstance(padding, str):
        if padding == "SAME":
            pad_specific = [
                _handle_padding(x_shape[i], strides[i], kernel[i], padding)
                for i in range(dims - 1, -1, -1)
            ]
            pad_list_top = [pad_specific[i] // 2 for i in range(dims)]
            pad_list_bot = [pad_specific[i] - pad_specific[i] // 2 for i in range(dims)]
            padding = [None] * len(pad_list_top) * 2
            padding[::2] = pad_list_top
            padding[1::2] = pad_list_bot
            pad_specific = pad_specific[::-1]
        else:
            pad_specific = [0] * dims
            padding = [0] * dims * 2
    else:
        if isinstance(padding, int):
            padding = [(padding, padding)] * dims
        pad_specific = [sum(padding[i]) for i in range(dims)]
        padding = [item for sublist in padding for item in sublist[::-1]][::-1]
    return padding, pad_specific


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def avg_pool1d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    if data_format in ("NWC", "NCL"):
        x = x.permute(0, 2, 1)

    if (
        isinstance(padding, int)
        or not isinstance(padding, str)
        and padding[0][0] == padding[0][1]
    ) and not divisor_override:
        if not isinstance(padding, int):
            padding = padding[0][0]
        res = torch.nn.functional.avg_pool1d(
            x,
            kernel,
            strides,
            padding,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
        )
    else:
        x_shape = x.shape[2]
        padding, pad_specific = _get_specific_pad(
            [x_shape], kernel, strides, padding, 1
        )
        x = torch.nn.functional.pad(x, padding, value=0.0)

        res = torch.nn.functional.avg_pool1d(x, kernel, strides, 0, ceil_mode)

        if not count_include_pad and any(pad_specific):
            num_padded_values = ivy.map(
                _get_num_padded_values,
                constant={
                    "p": pad_specific[0],
                    "n": x_shape,
                    "k": kernel[0],
                    "s": strides[0],
                },
                unique={
                    "i": torch.arange(res.shape[2]),
                },
            )
            num_padded_values = torch.tensor(num_padded_values, dtype=res.dtype)
            if ceil_mode:
                _, c = _padding_ceil_mode(x_shape, kernel[0], padding, strides[0], True)
                num_padded_values[-1] = _add_ceil_pad_to_pad_list(
                    num_padded_values[-1], kernel[0], c
                )
            res = (kernel[0] * res) / (kernel[0] - num_padded_values)

    if data_format in ("NWC", "NCL"):
        res = res.permute(0, 2, 1)

    return res


def _adjust_num_padded_values_to_ceil(
    pad_specific, num_padded_values, x_shape, kernel, strides, dims
):
    for i in range(dims):
        pad = [pad_specific[i] // 2, pad_specific[i] - pad_specific[i] // 2]
        _, c = _padding_ceil_mode(x_shape[i], kernel[i], pad, strides[i], True)
        num_padded_values[i][-1] = _add_ceil_pad_to_pad_list(
            num_padded_values[i][-1], kernel[i], c
        )
    return num_padded_values


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def avg_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    elif len(strides) == 1:
        kernel = (kernel[0], kernel[0])

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)

    if (
        isinstance(padding, int)
        or not isinstance(padding, str)
        and all(pad[0] == pad[1] for pad in padding)
    ):
        if not isinstance(padding, int):
            padding = [padding[0][0], padding[1][0]]
        res = torch.nn.functional.avg_pool2d(
            x,
            kernel,
            strides,
            padding,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
        )
    else:
        x_shape = list(x.shape[2:])
        padding, pad_specific = _get_specific_pad(x_shape, kernel, strides, padding, 2)
        x = torch.nn.functional.pad(
            x,
            padding,
            value=0.0,
        )
        res = torch.nn.functional.avg_pool2d(
            x, kernel, strides, 0, ceil_mode, divisor_override=divisor_override
        )

        if not count_include_pad and any(pad_specific) and not divisor_override:
            num_padded_values = [
                ivy.map(
                    _get_num_padded_values,
                    constant={
                        "p": pad_specific[i],
                        "n": x_shape[i],
                        "k": kernel[i],
                        "s": strides[i],
                    },
                    unique={
                        "i": torch.arange(res.shape[i + 2]),
                    },
                )
                for i in range(2)
            ]
            if ceil_mode:
                for i in range(2):
                    num_padded_values = _adjust_num_padded_values_to_ceil(
                        pad_specific, num_padded_values, x_shape, kernel, strides, 2
                    )
            num_padded_values1 = torch.tensor(num_padded_values[0], dtype=res.dtype)[
                :, None
            ]
            num_padded_values2 = torch.tensor(num_padded_values[1], dtype=res.dtype)[
                None, :
            ]
            num_padded_values = (
                num_padded_values1 * kernel[1]
                + num_padded_values2 * kernel[0]
                - num_padded_values1 * num_padded_values2
            )
            res = (kernel[0] * kernel[1] * res) / (
                kernel[0] * kernel[1] - num_padded_values
            )

    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def avg_pool3d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0], strides[0])
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    elif len(kernel) == 1:
        kernel = (kernel[0], kernel[0], kernel[0])
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)

    if (
        isinstance(padding, int)
        or not isinstance(padding, str)
        and all(pad[0] == pad[1] for pad in padding)
    ):
        if not isinstance(padding, int):
            padding = [padding[0][0], padding[1][0], padding[2][0]]
        res = torch.nn.functional.avg_pool3d(
            x,
            kernel,
            strides,
            padding,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            divisor_override=divisor_override,
        )
    else:
        x_shape = list(x.shape[2:])
        padding, pad_specific = _get_specific_pad(x_shape, kernel, strides, padding, 3)
        x = torch.nn.functional.pad(
            x,
            padding,
            value=0.0,
        )
        res = torch.nn.functional.avg_pool3d(
            x, kernel, strides, 0, ceil_mode, divisor_override=divisor_override
        )

        if not count_include_pad and any(pad_specific) and not divisor_override:
            num_padded_values = [
                torch.tensor(
                    ivy.map(
                        _get_num_padded_values,
                        constant={
                            "p": pad_specific[i],
                            "n": x_shape[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={
                            "i": torch.arange(res.shape[i + 2]),
                        },
                    ),
                    dtype=res.dtype,
                )
                for i in range(3)
            ]
            if ceil_mode:
                for i in range(3):
                    num_padded_values = _adjust_num_padded_values_to_ceil(
                        pad_specific, num_padded_values, x_shape, kernel, strides, 3
                    )
            num_padded_values1 = num_padded_values[0].reshape((-1, 1, 1))
            num_padded_values2 = num_padded_values[1].reshape((1, -1, 1))
            num_padded_values3 = num_padded_values[2].reshape((1, 1, -1))
            num_padded_values = (
                num_padded_values1 * kernel[1] * kernel[2]
                + num_padded_values2 * kernel[0] * kernel[2]
                + num_padded_values3 * kernel[0] * kernel[1]
                + num_padded_values1 * num_padded_values2 * num_padded_values3
                - num_padded_values1 * num_padded_values2 * kernel[2]
                - num_padded_values1 * num_padded_values3 * kernel[1]
                - num_padded_values2 * num_padded_values3 * kernel[0]
            )
            kernel_mul = kernel[0] * kernel[1] * kernel[2]
            res = (kernel_mul * res) / (kernel_mul - num_padded_values)

    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, backend_version)
def dct(
    x: torch.Tensor,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    if norm not in (None, "ortho"):
        raise ValueError("Norm must be either None or 'ortho'")
    if axis < 0:
        axis = axis + len(x.shape)
    if n is not None:
        signal_len = x.shape[axis]
        if n <= signal_len:
            local_idx = [slice(None)] * len(x.shape)
            local_idx[axis] = slice(None, n)
            x = x[local_idx]
        else:
            pad_idx = [0] * 2 * len(x.shape)
            pad_idx[(len(pad_idx) - 1) - (2 * axis)] = n - signal_len
            x = torch.nn.functional.pad(x, pad_idx)
    real_zero = torch.tensor(0.0, dtype=x.dtype)
    axis_dim = x.shape[axis]
    axis_dim_float = torch.tensor(axis_dim, dtype=x.dtype)

    if type == 1:
        if norm:
            raise ValueError("Normalization not supported for type-I DCT")
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(1, -1)
        x = torch.concat([x, x.flip(axis)[axis_idx]], dim=axis)
        dct_out = torch.real(torch.fft.rfft(x, dim=axis))
        return dct_out

    elif type == 2:
        scale_dims = [1] * len(x.shape)
        scale_dims[axis] = axis_dim
        complex_part = torch.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float
        scale = 2.0 * torch.exp(
            torch.complex(
                real_zero,
                -complex_part.type(real_zero.type()),
            )
        ).view(scale_dims)

        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(None, axis_dim)
        dct_out = torch.real(
            torch.fft.rfft(x, n=2 * axis_dim, axis=axis)[axis_idx] * scale
        )
        if norm == "ortho":
            n1 = 0.5 * torch.rsqrt(axis_dim_float)
            n2 = n1 * math.sqrt(2.0)
            sf = torch.nn.functional.pad(n1.unsqueeze(0), (0, axis_dim - 1), value=n2)
            dct_out = sf.view(scale_dims) * dct_out
        return dct_out

    elif type == 3:
        scale_dims = [1] * len(x.shape)
        scale_dims[axis] = axis_dim
        complex_part = torch.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float
        scale = 2.0 * torch.exp(
            torch.complex(real_zero, complex_part.type(real_zero.type()))
        ).view(scale_dims)
        if norm == "ortho":
            n1 = torch.sqrt(axis_dim_float)
            n2 = n1 * math.sqrt(0.5)
            scale_dims = [1] * len(x.shape)
            scale_dims[axis] = axis_dim
            sf = torch.nn.functional.pad(n1.unsqueeze(0), (0, axis_dim - 1), value=n2)
            x = x * sf.view(scale_dims)
        else:
            x = x * axis_dim_float

        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(None, axis_dim)
        dct_out = torch.real(
            torch.fft.irfft(
                scale * torch.complex(x, real_zero), n=2 * axis_dim, axis=axis
            )
        )[axis_idx]
        return dct_out

    elif type == 4:
        dct_2 = dct(x, type=2, n=2 * axis_dim, axis=axis, norm=None)
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(1, None, 2)
        dct_out = dct_2[axis_idx]
        if norm == "ortho":
            dct_out *= math.sqrt(0.5) * torch.rsqrt(axis_dim_float)
        return dct_out


def idct(
    x: torch.Tensor,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.tensor:
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return dct(x, type=inverse_type, n=n, axis=axis, norm=norm, out=out)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def fft(
    x: torch.Tensor,
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    if x.dtype in [torch.int64, torch.float64, torch.complex128]:
        out_dtype = torch.complex128
    else:
        out_dtype = torch.complex64
    return torch.fft.fft(x, n, dim, norm, out=out).to(dtype=out_dtype)


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
def dropout(
    x: torch.Tensor,
    prob: float,
    /,
    *,
    scale: bool = True,
    dtype: torch.dtype = None,
    training: bool = True,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence[int]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = ivy.astype(x, dtype) if dtype and x.dtype != dtype else x
    if prob == 0 or not training:
        return x
    res = torch.nn.functional.dropout(x, prob, training=True)
    res = torch.multiply(res, (1.0 - prob)) if not scale else res
    return res


dropout.partial_mixed_handler = lambda x, prob, **kwargs: (
    kwargs.get("noise_shape") is None and kwargs.get("seed") is None
)


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    backend_version,
)
def dropout1d(
    x: torch.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_batched = len(x.shape) == 3
    if data_format == "NWC":
        perm = (0, 2, 1) if is_batched else (1, 0)
        x = torch.permute(x, perm)
    res = torch.nn.functional.dropout1d(x, prob, training=training)
    if data_format == "NWC":
        res = torch.permute(res, perm)
    return res


@with_unsupported_dtypes(
    {"2.2 and below": ("float16",)},
    backend_version,
)
def dropout2d(
    x: torch.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_batched = len(x.shape) == 4
    if data_format == "NHWC":
        perm = (0, 3, 1, 2) if is_batched else (2, 0, 1)
        x = torch.permute(x, perm)
    res = torch.nn.functional.dropout2d(x, prob, training=training)
    if data_format == "NHWC":
        perm = (0, 2, 3, 1) if is_batched else (1, 2, 0)
        res = torch.permute(res, perm)
    return res


@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def dropout3d(
    x: torch.Tensor,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    is_batched = len(x.shape) == 5
    if data_format == "NDHWC":
        perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
        x = torch.permute(x, perm)
    res = torch.nn.functional.dropout3d(x, prob, training=training)
    if data_format == "NDHWC":
        perm = (0, 2, 3, 4, 1) if is_batched else (1, 2, 3, 0)
        res = torch.permute(res, perm)
    return res


def ifft(
    x: torch.Tensor,
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return torch.fft.ifft(x, n, dim, norm, out=out).resolve_conj()


@with_unsupported_dtypes({"2.2 and below": ("complex",)}, backend_version)
def embedding(
    weights: torch.Tensor,
    indices: torch.Tensor,
    /,
    *,
    max_norm: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ivy.utils.assertions.check_equal(
        len(weights.shape), 2, message="weights must be 2-d", as_array=False
    )
    return torch.nn.functional.embedding(indices, weights, max_norm=max_norm)


embedding.support_native_out = False


def interpolate(
    x: torch.Tensor,
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "tf_bicubic",
        "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: bool = False,
    antialias: bool = False,
    out: Optional[torch.Tensor] = None,
):
    if mode not in ["linear", "bilinear", "bicubic", "trilinear"]:
        align_corners = None
    return torch.nn.functional.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute_scale_factor,
    )


interpolate.partial_mixed_handler = (
    lambda *args, mode="linear", align_corners=False, **kwargs: mode
    not in [
        "tf_area",
        "nd",
        "tf_bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ]
    and (mode in ["linear", "bilinear", "bicubic", "trilinear"] or not align_corners)
)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def adaptive_max_pool2d(
    input: torch.Tensor, output_size: Union[Sequence[int], int]
) -> torch.Tensor:
    return torch.nn.functional.adaptive_max_pool2d(input, output_size)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def adaptive_max_pool3d(
    input: torch.Tensor, output_size: Union[Sequence[int], int]
) -> torch.Tensor:
    return torch.nn.functional.adaptive_max_pool3d(input, output_size)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def adaptive_avg_pool1d(input, output_size):
    return torch.nn.functional.adaptive_avg_pool1d(input, output_size)


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def adaptive_avg_pool2d(input, output_size, /, *, data_format: str = "NHWC"):
    squeeze = False
    if input.ndim == 3:
        input = torch.unsqueeze(input, 0)
        squeeze = True
    permuted_input = False
    if data_format == "NHWC":
        input = torch.permute(input, (0, input.ndim - 1, *range(1, input.ndim - 1)))
        permuted_input = True
    ret = torch.nn.functional.adaptive_avg_pool2d(input, output_size)
    ret = torch.permute(ret, (0, *range(2, input.ndim), 1)) if permuted_input else ret
    ret = torch.squeeze(ret, 0) if squeeze else ret
    return ret


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def fft2(
    x: torch.Tensor,
    *,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not all(isinstance(j, int) for j in dim):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {dim} to be a sequence of integers <class integer>"
        )
    if s is None:
        s = (x.shape[dim[0]], x.shape[dim[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return torch.tensor(
        torch.fft.fft2(x, s, dim, norm, out=out), dtype=torch.complex128
    )


def ifftn(
    x: torch.Tensor,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fft.ifftn(x, s=s, dim=axes, norm=norm, out=out)


def rfft(
    x: torch.Tensor,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x.real
    if x.dtype == torch.float16:
        x = x.to(torch.float32)

    ret = torch.fft.rfft(x, n=n, dim=axis, norm=norm)

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, backend_version)
def rfftn(
    x: torch.Tensor,
    s: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
    *,
    norm: str = "backward",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not all(isinstance(j, int) for j in axes):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {axes} to be a sequence of integers <class integer>"
        )
    if s is None:
        s = (x.shape[axes[0]], x.shape[axes[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid axes {axes}, expecting ranging"
            f" from {-len(x.shape)} to {len(x.shape)-1}"
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return torch.tensor(
        torch.fft.rfftn(x, s, axes, norm=norm, out=out), dtype=torch.complex128
    )


# stft
@with_unsupported_dtypes(
    {
        "2.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def stft(
    signals: torch.Tensor,
    frame_length: int,
    frame_step: int,
    /,
    *,
    fft_length: Optional[int] = None,
    window_fn: Optional[Callable] = None,
    pad_end: Optional[bool] = False,
    name: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(frame_length, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_length)}"
        )

    if frame_length < 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {frame_length}, expecting frame_length larger than or"
            " equal to 1"
        )

    if not isinstance(frame_step, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_step)}"
        )

    if frame_step < 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {frame_length}, expecting frame_length larger than or"
            " equal to 1"
        )

    if fft_length is not None:
        if not isinstance(fft_length, int):
            raise ivy.utils.exceptions.IvyError(
                f"Expecting <class 'int'> instead of {type(fft_length)}"
            )

        if fft_length < 1:
            raise ivy.utils.exceptions.IvyError(
                f"Invalid data points {frame_length}, expecting frame_length larger"
                " than or equal to 1"
            )

    input_dtype = signals.dtype
    if input_dtype == torch.float32:
        dtype = torch.complex64
    elif input_dtype == torch.float64:
        dtype = torch.complex128

    def stft_1D(signals, frame_length, frame_step, fft_length, pad_end):
        if fft_length is None:
            fft_length = 1
            while fft_length < frame_length:
                fft_length *= 2

        num_samples = signals.shape[-1]

        if pad_end:
            num_samples = signals.shape[-1]
            num_frames = -(-num_samples // frame_step)
            pad_length = max(
                0, frame_length + frame_step * (num_frames - 1) - num_samples
            )

            signals = torch.nn.functional.pad(signals, (0, pad_length))
        else:
            num_frames = 1 + (num_samples - frame_length) // frame_step

        stft_result = []

        if window_fn is None:
            window = 1
        else:
            window = window_fn(frame_length)

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_length
            frame = signals[..., start:end]
            windowed_frame = frame * window
            pad_length = fft_length - frame_length
            windowed_frame = torch.nn.functional.pad(windowed_frame, (0, pad_length))
            windowed_frame = torch.tensor(windowed_frame, dtype=dtype)

            fft_frame = torch.fft.fft(windowed_frame, axis=-1)
            slit = int(fft_length // 2 + 1)
            stft_result.append(fft_frame[..., 0:slit])

        stft = torch.stack(stft_result, axis=0)
        return stft

    def stft_helper(nested_list, frame_length, frame_step, fft_length):
        nested_list = nested_list
        if len(nested_list.shape) > 1:
            return [
                stft_helper(sublist, frame_length, frame_step, fft_length)
                for sublist in nested_list
            ]
        else:
            return stft_1D(nested_list, frame_length, frame_step, fft_length, pad_end)

    to_return = stft_helper(signals, frame_length, frame_step, fft_length)
    flat_list = [
        item if isinstance(item, torch.Tensor) else torch.tensor(item)
        for sublist in to_return
        for item in sublist
    ]
    result = torch.stack(flat_list)
    original_shape = (len(to_return), len(to_return[0]))
    result = result.view(original_shape + result.shape[1:])
    return result


def sliding_window(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    /,
    *,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
) -> torch.Tensor:
    if input.ndim != 4:
        # convert input to 4D tensor as unfold only accepts 4D data
        input_shape = input.shape
        extend_dims = max(0, 4 - len(input_shape))
        new_shape = (1,) * extend_dims + input_shape
        input = input.reshape(new_shape).float()

    stride = (stride,) * 2 if isinstance(stride, int) else tuple(stride) * 2
    dilation = (dilation,) * 2 if isinstance(dilation, int) else tuple(dilation) * 2

    kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
    if len(kernel_size) < 2:
        kernel_size = (kernel_size) * 2

    # check padding and convert to right format
    if isinstance(padding, str):
        # convert padding from str to seq
        if padding.upper() == "SAME":
            pad_vals = []
            for dim in input.shape:
                pad_val = _handle_padding(
                    dim,
                    stride[0] if isinstance(stride, tuple) else stride,
                    kernel_size[0],
                    padding,
                )
                pad_vals.append(pad_val)
            padding = pad_vals[:2]
        else:
            padding = 0
    else:
        padding = (padding,) * 2 if isinstance(padding, int) else padding

    return torch.nn.functional.unfold(
        input, kernel_size, dilation=dilation, padding=padding, stride=stride
    )


def max_unpool1d(
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size: Union[Tuple[int], int],
    /,
    *,
    strides: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    data_format: Optional[str] = "NCW",
) -> torch.Tensor:
    if strides is None:
        strides = kernel_size
    revert = False
    if data_format in ["NCW", "NWC"]:
        if data_format == "NWC":
            input = input.permute(0, 2, 1)
            indices = indices.permute(0, 2, 1)
            revert = True
    else:
        raise ValueError(
            f"data_format attr should be NCW or NWC but found {data_format}"
        )
    kernel_size = _broadcast_pooling_helper(kernel_size, "1d", name="kernel_size")
    padding = _broadcast_pooling_helper(padding, "1d", name="padding")
    strides = _broadcast_pooling_helper(strides, "1d", name="strides")
    ret = torch.nn.functional.max_unpool1d(
        input,
        indices,
        kernel_size,
        strides,
        padding,
    )
    if revert:
        ret = ret.permute(0, 2, 1)
    return ret


def _max_unpool1d_mixed_handler(input, indices, kernel_size, **kwargs):
    dt = kwargs.get("data_format", "NCW")
    inds = indices.permute(0, 2, 1) if dt == "NWC" else indices
    flat_inds = inds.reshape((-1,))
    stride = indices.shape[-1]
    not_dup = True
    for i in range(0, flat_inds.numel(), stride):
        inds = flat_inds[i : (i + stride)]
        inds = inds.unique()
        if inds.numel() != stride:
            not_dup = False
    return not_dup


max_unpool1d.partial_mixed_handler = _max_unpool1d_mixed_handler
