"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

from typing import List, Optional, Tuple, Union, Sequence

# global
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _out_shape(x, strides, pad, dilations, filters):
    return (x - 1) * strides - 2 * pad + dilations * (filters - 1) + 1


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: int,
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, tuple):
        strides = strides[0]
    if isinstance(dilations, tuple):
        dilations = dilations[0]
    f_w_after_dilation = filters.shape[0] + ((dilations - 1) * (filters.shape[0] - 1))
    filters = filters.permute(2, 1, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x_shape = x.shape[2]
    pad_w = ivy.handle_padding(x_shape, strides, f_w_after_dilation, padding)
    x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2], value=0)
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv1d(x, filters, None, strides, "valid", dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x,
    filters,
    strides: int,
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[torch.Tensor] = None,
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(1, 2, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if output_shape is None:
        new_w = ivy.deconv_length(
            x.shape[2], strides, filter_shape[0], padding, dilations
        )
        output_shape = [x.shape[0], new_w, filters.shape[-2]]
    elif len(output_shape) == 1:
        output_shape = [x.shape[0], output_shape[0], filters.shape[-2]]
    not_valid_h = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
    pad_w = ivy.handle_padding(output_shape[1], strides, filter_shape[0], padding)
    if padding == "VALID":
        padding_list: List[int] = [0]
    elif padding == "SAME":
        if pad_w % 2 != 0:
            pad_w -= 1
            not_valid_h = True
        pad_w = pad_w // 2
        padding_list = [pad_w]
    else:
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_w = _out_shape(x.shape[2], strides, pad_w, dilations, filters.shape[2])
    output_padding = [max(output_shape[1] - out_w, 0)]
    res = torch.nn.functional.conv_transpose1d(
        x,
        filters,
        None,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_h:
        res = res[:, :, 0:-1]
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if isinstance(dilations, int):
        dilations = (dilations, dilations)
    elif len(dilations) == 1:
        dilations = (dilations[0], dilations[0])

    f_w_after_dilation = filters.shape[1] + (
        (dilations[1] - 1) * (filters.shape[1] - 1)
    )
    f_h_after_dilation = filters.shape[0] + (
        (dilations[0] - 1) * (filters.shape[0] - 1)
    )
    filter_shape = [f_h_after_dilation, f_w_after_dilation]
    filters = filters.permute(3, 2, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv2d(x, filters, None, strides, "valid", dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
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
    output_shape=None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[torch.Tensor] = None,
):
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filter_shape = list(filters.shape[0:2])
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if output_shape is None:
        new_h = ivy.deconv_length(
            x.shape[2], strides[0], filter_shape[0], padding, dilations[0]
        )
        new_w = ivy.deconv_length(
            x.shape[3], strides[1], filter_shape[1], padding, dilations[1]
        )
        output_shape = [x.shape[0], new_h, new_w, filters.shape[1]]
    elif len(output_shape) == 2:
        output_shape = [x.shape[0]] + output_shape + [filters.shape[1]]
    not_valid_h = False
    not_valid_w = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations[0] - 1)
    filter_shape[1] = filter_shape[1] + (filter_shape[1] - 1) * (dilations[1] - 1)
    pad_h = ivy.handle_padding(output_shape[2], strides[0], filter_shape[0], padding)
    pad_w = ivy.handle_padding(output_shape[1], strides[1], filter_shape[1], padding)
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
    elif padding == "SAME":
        if pad_h % 2 != 0:
            pad_h -= 1
            not_valid_h = True
        if pad_w % 2 != 0:
            pad_w -= 1
            not_valid_w = True
        pad_h = pad_h // 2
        pad_w = pad_w // 2

        padding_list = [pad_h, pad_w]

    else:
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_h = _out_shape(x.shape[2], strides[0], pad_h, dilations[0], filters.shape[2])
    out_w = _out_shape(x.shape[3], strides[1], pad_w, dilations[1], filters.shape[3])
    output_padding = [
        max(output_shape[1] - out_h, 0),
        max(output_shape[2] - out_w, 0),
    ]
    res = torch.nn.functional.conv_transpose2d(
        x,
        filters,
        None,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_h:
        res = res[:, :, 0:-1, :]
    if not_valid_w:
        res = res[:, :, :, 0:-1]
    if data_format == "NHWC":
        res = res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations

    f_w_after_dilation = filters.shape[1] + (
        (dilations[1] - 1) * (filters.shape[1] - 1)
    )
    f_h_after_dilation = filters.shape[0] + (
        (dilations[0] - 1) * (filters.shape[0] - 1)
    )
    filter_shape = [f_h_after_dilation, f_w_after_dilation]
    dims_in = filters.shape[-1]
    filters = torch.unsqueeze(filters, -1)
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
    )

    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    # noinspection PyArgumentEqualDefault
    res = torch.nn.functional.conv2d(
        x, filters, None, strides, "valid", dilations, dims_in
    )
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
# noinspection PyUnresolvedReferences
def conv3d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    out: Optional[torch.Tensor] = None,
):
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    f_w_after_dilation = filters.shape[2] + (
        (dilations[2] - 1) * (filters.shape[2] - 1)
    )
    f_h_after_dilation = filters.shape[1] + (
        (dilations[1] - 1) * (filters.shape[1] - 1)
    )
    f_d_after_dilation = filters.shape[0] + (
        (dilations[0] - 1) * (filters.shape[0] - 1)
    )
    filter_shape = [f_d_after_dilation, f_h_after_dilation, f_w_after_dilation]
    filters = filters.permute(4, 3, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    x_shape = list(x.shape[2:])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], filter_shape[2], padding)
    x = torch.nn.functional.pad(
        x,
        [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        ],
        value=0,
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv3d(x, filters, None, strides, "valid", dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
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
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    filter_shape = list(filters.shape[0:3])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if output_shape is None:
        new_d = ivy.deconv_length(
            x.shape[2], strides[0], filter_shape[0], padding, dilations[0]
        )
        new_h = ivy.deconv_length(
            x.shape[3], strides[1], filter_shape[1], padding, dilations[1]
        )
        new_w = ivy.deconv_length(
            x.shape[4], strides[2], filter_shape[2], padding, dilations[2]
        )
        output_shape = [x.shape[0], new_d, new_h, new_w, filters.shape[1]]
    elif len(output_shape) == 3:
        output_shape = [
            x.shape[0],
            output_shape[0],
            output_shape[1],
            output_shape[2],
            filters.shape[1],
        ]
    not_valid_h = False
    not_valid_d = False
    not_valid_w = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations[0] - 1)
    filter_shape[1] = filter_shape[1] + (filter_shape[1] - 1) * (dilations[1] - 1)
    filter_shape[2] = filter_shape[2] + (filter_shape[2] - 1) * (dilations[2] - 1)
    pad_d = ivy.handle_padding(output_shape[1], strides[0], filter_shape[0], padding)
    pad_h = ivy.handle_padding(output_shape[2], strides[1], filter_shape[1], padding)
    pad_w = ivy.handle_padding(output_shape[3], strides[2], filter_shape[2], padding)
    if padding == "VALID":
        padding_list: List[int] = [0, 0, 0]
    elif padding == "SAME":
        if pad_d % 2 != 0:
            pad_d -= 1
            not_valid_d = True
        if pad_h % 2 != 0:
            pad_h -= 1
            not_valid_h = True
        if pad_w % 2 != 0:
            pad_w -= 1
            not_valid_w = True
        pad_d = pad_d // 2
        pad_h = pad_h // 2
        pad_w = pad_w // 2
        padding_list = [pad_d, pad_h, pad_w]

    else:
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_d = _out_shape(x.shape[2], strides[0], pad_d, dilations[0], filters.shape[2])
    out_h = _out_shape(x.shape[3], strides[1], pad_h, dilations[1], filters.shape[3])
    out_w = _out_shape(x.shape[4], strides[2], pad_w, dilations[2], filters.shape[4])
    output_padding = [
        max(output_shape[1] - out_d, 0),
        max(output_shape[2] - out_h, 0),
        max(output_shape[3] - out_w, 0),
    ]
    res = torch.nn.functional.conv_transpose3d(
        x,
        filters,
        None,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding,
    )
    if not_valid_d:
        res = res[:, :, 0:-1, :, :]
    if not_valid_h:
        res = res[:, :, :, 0:-1, :]
    if not_valid_w:
        res = res[:, :, :, :, 0:-1]
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def conv_general_dilated(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    out: Optional[torch.Tensor] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations

    filter_shape = [
        filters.shape[i] + (filters.shape[i] - 1) * (dilations[i] - 1)
        for i in range(dims)
    ]
    filters = filters.permute(-1, -2, *range(dims))
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    x_shape = x.shape[2:]
    # adding dilation to input
    if dims == 1:
        permute_list = [2]
    else:
        permute_list = [i for i in range(3, dims + 2)]
        permute_list += [2]
    for i in range(dims):
        if x_dilations[i] > 1:
            new_x = x_shape[i] + (x_shape[i] - 1) * (x_dilations[i] - 1)
            h = torch.eye(new_x, dtype=x.dtype)[:: x_dilations[i]]
            x = torch.einsum("...kl, lm -> ...km", x.permute(0, 1, *permute_list), h)

    x_shape = list(x.shape[2:])
    pad_specific = [
        ivy.handle_padding(x_shape[i], strides[i], filter_shape[i], padding)
        for i in range(dims - 1, -1, -1)
    ]
    pad_list_top = [pad_specific[i] // 2 for i in range(dims)]
    pad_list_bot = [pad_specific[i] - pad_specific[i] // 2 for i in range(dims)]
    pad_list = [None] * len(pad_list_top) * 2
    pad_list[::2] = pad_list_top
    pad_list[1::2] = pad_list_bot
    x = torch.nn.functional.pad(
        x,
        pad_list,
        value=0,
    )
    if dims == 1:
        res = torch.nn.functional.conv1d(
            x, filters, None, strides, "valid", dilations, feature_group_count
        )
    elif dims == 2:
        res = torch.nn.functional.conv2d(
            x, filters, None, strides, "valid", dilations, feature_group_count
        )
    else:
        res = torch.nn.functional.conv3d(
            x, filters, None, strides, "valid", dilations, feature_group_count
        )
    if data_format == "channel_last":
        return res.permute(0, *range(2, dims + 2), 1)
    return res


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def conv_general_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape=None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    out: Optional[torch.Tensor] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    filter_shape = list(filters.shape[0:dims])
    filters = filters.permute(dims, dims + 1, *range(dims))
    if data_format == "channel_last":
        x = x.permute(0, dims + 1, *range(1, dims + 1))
    if output_shape is None:
        out_shape = [
            ivy.deconv_length(
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
        ivy.handle_padding(output_shape[i + 1], strides[i], filter_shape[i], padding)
        for i in range(dims)
    ]
    if padding == "VALID":
        padding_list = [0] * dims
    elif padding == "SAME":
        for i in range(dims):
            if pad_specific[i] % 2 != 0:
                pad_specific[i] -= 1
                not_valid_pad[i] = True
        padding_list = [pad_specific[i] // 2 for i in range(dims)]
    out_shape = [
        _out_shape(
            x.shape[i + 2],
            strides[i],
            padding_list[i],
            dilations[i],
            filters.shape[i + 2],
        )
        for i in range(dims)
    ]
    output_padding = [max(output_shape[i + 1] - out_shape[i], 0) for i in range(dims)]
    if dims == 1:
        res = torch.nn.functional.conv_transpose1d(
            x,
            filters,
            None,
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
            None,
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
            None,
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
