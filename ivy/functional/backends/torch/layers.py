"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import List, Optional, Tuple, Union, Sequence

# local
import ivy


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding == "VALID":
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == "SAME":
        dim_size = dim_size * stride_size

    return dim_size


def _out_shape(x, strides, pad, dilations, filters):
    return (x - 1) * strides - 2 * pad + dilations * (filters - 1) + 1


def _handle_padding(x, strides, filters, padding):
    if padding == "SAME":
        if x % strides == 0:
            pad = max(filters - strides, 0)
        else:
            pad = max(filters - (x % strides), 0)
    else:
        pad = 0
    return pad


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
    pad_w = _handle_padding(x_shape, strides, f_w_after_dilation, padding)
    x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2], value=0)
    if padding != "VALID" and padding != "SAME":
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv1d(x, filters, None, strides, "valid", dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


conv1d.unsupported_dtypes = ("float16",)


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
    filters = filters.permute(2, 1, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if output_shape is None:
        new_w = _deconv_length(x.shape[2], strides, filter_shape[0], padding, dilations)
        output_shape = [new_w]
    not_valid_h = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations - 1)
    pad_w = _handle_padding(output_shape[0], strides, filter_shape[0], padding)
    if padding == "VALID":
        padding_list: List[int] = [0]
    elif padding == "SAME":
        if pad_w % 2 != 0:
            pad_w -= 1
            not_valid_h = True
        pad_w = pad_w // 2
        padding_list = [pad_w]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_w = _out_shape(x.shape[2], strides, pad_w, dilations, filters.shape[2])
    output_padding = [max(output_shape[0] - out_w, 0)]
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


conv1d_transpose.unsupported_dtypes = ("float16",)


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
    pad_h = _handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_w = _handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
    )
    if padding != "VALID" and padding != "SAME":
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv2d(x, filters, None, strides, "valid", dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


conv2d.unsupported_dtypes = ("float16",)


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
    filters = filters.permute(3, 2, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if output_shape is None:
        new_h = _deconv_length(
            x.shape[2], strides[0], filter_shape[0], padding, dilations[0]
        )
        new_w = _deconv_length(
            x.shape[3], strides[1], filter_shape[1], padding, dilations[1]
        )
        output_shape = [new_h, new_w]
    not_valid_h = False
    not_valid_w = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations[0] - 1)
    filter_shape[1] = filter_shape[1] + (filter_shape[1] - 1) * (dilations[1] - 1)
    pad_w = _handle_padding(output_shape[1], strides[1], filter_shape[1], padding)
    pad_h = _handle_padding(output_shape[0], strides[0], filter_shape[0], padding)
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
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_h = _out_shape(x.shape[2], strides[0], pad_h, dilations[0], filters.shape[2])
    out_w = _out_shape(x.shape[3], strides[1], pad_w, dilations[1], filters.shape[3])
    output_padding = [
        max(output_shape[0] - out_h, 0),
        max(output_shape[1] - out_w, 0),
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


conv2d_transpose.unsupported_dtypes = ("float16",)


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
    pad_h = _handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_w = _handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
    )

    if padding != "VALID" and padding != "SAME":
        raise Exception(
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


depthwise_conv2d.unsupported_dtypes = ("float16",)


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
    pad_d = _handle_padding(x_shape[0], strides[0], filter_shape[0], padding)
    pad_h = _handle_padding(x_shape[1], strides[1], filter_shape[1], padding)
    pad_w = _handle_padding(x_shape[2], strides[2], filter_shape[2], padding)
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
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv3d(x, filters, None, strides, "valid", dilations)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


conv3d.unsupported_dtypes = ("float16",)


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
        new_d = _deconv_length(
            x.shape[2], strides[0], filter_shape[0], padding, dilations[0]
        )
        new_h = _deconv_length(
            x.shape[3], strides[1], filter_shape[1], padding, dilations[1]
        )
        new_w = _deconv_length(
            x.shape[4], strides[2], filter_shape[2], padding, dilations[2]
        )
        output_shape = [new_d, new_h, new_w]
    not_valid_h = False
    not_valid_d = False
    not_valid_w = False
    filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations[0] - 1)
    filter_shape[1] = filter_shape[1] + (filter_shape[1] - 1) * (dilations[1] - 1)
    filter_shape[2] = filter_shape[2] + (filter_shape[2] - 1) * (dilations[2] - 1)
    pad_d = _handle_padding(output_shape[0], strides[0], filter_shape[0], padding)
    pad_h = _handle_padding(output_shape[1], strides[1], filter_shape[1], padding)
    pad_w = _handle_padding(output_shape[2], strides[2], filter_shape[2], padding)
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
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    out_d = _out_shape(x.shape[2], strides[0], pad_d, dilations[0], filters.shape[2])
    out_h = _out_shape(x.shape[3], strides[1], pad_h, dilations[1], filters.shape[3])
    out_w = _out_shape(x.shape[4], strides[2], pad_w, dilations[2], filters.shape[4])
    output_padding = [
        max(output_shape[0] - out_d, 0),
        max(output_shape[1] - out_h, 0),
        max(output_shape[2] - out_w, 0),
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


conv3d_transpose.unsupported_dtypes = ("float16",)
