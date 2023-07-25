"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""
import pdb
from typing import Optional, Tuple, Union, Sequence

# global
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ivy.functional.ivy.layers import _handle_padding, _deconv_length


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    /,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


linear.partial_mixed_handler = lambda x, weight, **kwargs: weight.ndim == 2

def _ff_xd_before_conv(x,filters,dims,filter_format,x_dilations):
    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))


    # adding dilation to input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    for i in range(dims):
        if x_dilations[i] > 1:
            h = x.shape[2 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = torch.eye(
                new_height,
                dtype=x.dtype,
                device=ivy.as_native_dev(ivy.default_device()),
            )[:: x_dilations[i]]
            x = torch.swapaxes(x, 2 + i, -1)
            x = torch.matmul(x, h)
            x = torch.swapaxes(x, -1, 2 + i)
    return x, filters





def _pad_before_conv(
    x, filters, strides, padding, dims, dilations, filter_format="channel_last"
):
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    filter_shape = (
        filters.shape[2:] if filter_format == "channel_first" else filters.shape[:dims]
    )
    if isinstance(padding, str):
        # use torch's padding in conv if strides are all 1
        if len(strides) == strides.count(1):
            return x, padding.lower()
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        pad_specific = [
            _handle_padding(x.shape[2 + i], strides[i], filter_shape[i], padding)
            for i in range(dims - 1, -1, -1)
        ]
        pad_list_top = [pad_specific[i] // 2 for i in range(dims)]
        pad_list_bot = [pad_specific[i] - pad_specific[i] // 2 for i in range(dims)]
        pad_list = [None] * len(pad_list_top) * 2
        pad_list[::2] = pad_list_top
        pad_list[1::2] = pad_list_bot
    else:
        if isinstance(padding, int):
            return x, padding
        # if symmetric padding is used, use torch's padding in conv function
        if all(pad[0] == pad[1] for pad in padding):
            return x, [pad[0] for pad in padding]
        pad_list = [item for sublist in padding for item in sublist[::-1]][::-1]
    return torch.nn.functional.pad(x, pad_list), "valid"


def _pad_before_conv_tranpose(
    x, filters, strides, padding, dims, dilations, output_shape, filter_shape
):
    if output_shape is None:
        out_shape = [
            _deconv_length(
                x.shape[i + 2], strides[i], filter_shape[i], padding, dilations[i]
            )
            for i in range(dims)
        ]
        output_shape = [x.shape[0], *out_shape, filters.shape[1]]
    elif len(output_shape) == dims:
        output_shape = [x.shape[0]] + output_shape + [filters.shape[1]]
    not_valid_pad = [False] * dims
    filter_shape = [
        filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
        for i in range(dims)
    ]
    pad_specific = [
        _handle_padding(output_shape[i + 1], strides[i], filter_shape[i], padding)
        for i in range(dims)
    ]
    if padding == "VALID":
        padding_list = [0] * dims
    else:
        for i in range(dims):
            if pad_specific[i] % 2 != 0:
                pad_specific[i] -= 1
                not_valid_pad[i] = True
        padding_list = [pad_specific[i] // 2 for i in range(dims)]
    out_shape = [
        (x.shape[i + 2] - 1) * strides[i]
        - 2 * padding_list[i]
        + dilations[i] * (filters.shape[i + 2] - 1)
        + 1
        for i in range(dims)
    ]
    output_padding = [max(output_shape[i + 1] - out_shape[i], 0) for i in range(dims)]
    return not_valid_pad, padding_list, output_padding


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x,filters =_ff_xd_before_conv(x,filters,1,filter_format,x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 1, dilations, "channel_first"
    )
    res = torch.nn.functional.conv1d(x, filters, bias, strides, padding, dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x, filters = _ff_xd_before_conv(x, filters, 1, filter_format, x_dilations)
    strides = [strides] if isinstance(strides, int) else strides
    dilations = [dilations] if isinstance(dilations, int) else dilations
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 1, dilations, output_shape, filters.shape[2:]
    )
    res = torch.nn.functional.conv_transpose1d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1]
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x,filters =_ff_xd_before_conv(x,filters,2,filter_format,x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    res = torch.nn.functional.conv2d(x, filters, bias, strides, padding, dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None
):
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)

    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    x, filters = _ff_xd_before_conv(x, filters, 2, filter_format, x_dilations)
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 2, dilations, output_shape, filters.shape[2:]
    )




    res = torch.nn.functional.conv_transpose2d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[..., :-1, :]
    if not_valid_pad[1]:
        res = res[..., :-1]
    if data_format == "NHWC":
        res = res.permute(0, *range(2, 4), 1)

    return res


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "bfloat16",
            "complex",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    filters = ivy.squeeze(filters, 3).to_native() if filters.ndim == 4 else filters
    filters = torch.unsqueeze(filters, -1)
    dims_in = filters.shape[-2]
    filters = filters.permute(2, 3, 0, 1)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 2, dilations, "channel_first"
    )
    # noinspection PyArgumentEqualDefault
    res = torch.nn.functional.conv2d(
        x, filters, None, strides, padding, dilations, dims_in
    )
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")}, backend_version
)
# noinspection PyUnresolvedReferences
def conv3d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int,int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    x,filters =_ff_xd_before_conv(x,filters,3,filter_format,x_dilations)
    x, padding = _pad_before_conv(
        x, filters, strides, padding, 3, dilations, "channel_first"
    )
    res = torch.nn.functional.conv3d(x, filters, bias, strides, padding, dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int,int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    x, filters = _ff_xd_before_conv(x, filters, 3, filter_format, x_dilations)
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, 3, dilations, output_shape, filters.shape[2:]
    )
    res = torch.nn.functional.conv_transpose3d(
        x,
        filters,
        bias,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_pad[0]:
        res = res[:, :, 0:-1, :, :]
    if not_valid_pad[1]:
        res = res[:, :, :, 0:-1, :]
    if not_valid_pad[2]:
        res = res[:, :, :, :, 0:-1]
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_dilated(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):
    # permuting dims based on formats
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))

    if filter_format == "channel_last":
        filters = filters.permute(-1, -2, *range(dims))

    # adding dilation to input
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    for i in range(dims):
        if x_dilations[i] > 1:
            h = x.shape[2 + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = torch.eye(
                new_height,
                dtype=x.dtype,
                device=ivy.as_native_dev(ivy.default_device()),
            )[:: x_dilations[i]]
            x = torch.swapaxes(x, 2 + i, -1)
            x = torch.matmul(x, h)
            x = torch.swapaxes(x, -1, 2 + i)

    x, padding = _pad_before_conv(
        x, filters, strides, padding, dims, dilations, "channel_first"
    )

    if dims == 1:
        res = torch.nn.functional.conv1d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    elif dims == 2:
        res = torch.nn.functional.conv2d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    else:
        res = torch.nn.functional.conv3d(
            x, filters, bias, strides, padding, dilations, feature_group_count
        )
    if data_format == "channel_last":
        return res.permute(0, *range(2, dims + 2), 1)
    return res


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv_general_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
):

    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    filters = filters.permute(dims, dims + 1, *range(dims))
    not_valid_pad, padding_list, output_padding = _pad_before_conv_tranpose(
        x, filters, strides, padding, dims, dilations, output_shape, filters.shape[2:]
    )
    if dims == 1:
        res = torch.nn.functional.conv_transpose1d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[:, :, :-1]
    elif dims == 2:
        res = torch.nn.functional.conv_transpose2d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :]
        if not_valid_pad[1]:
            res = res[..., :-1]
    else:
        res = torch.nn.functional.conv_transpose3d(
            x,
            filters,
            bias,
            strides,
            padding_list,
            dilation=dilations,
            output_padding=output_padding,
            groups=feature_group_count,
        )
        if not_valid_pad[0]:
            res = res[..., :-1, :, :]
        if not_valid_pad[1]:
            res = res[..., :, :-1, :]
        if not_valid_pad[2]:
            res = res[..., :, :, :-1]
    if data_format == "channel_last":
        res = res.permute(0, *range(2, dims + 2), 1)
    return res
