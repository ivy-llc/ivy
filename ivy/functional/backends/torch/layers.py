"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

# global
import math
import torch
from typing import List, Optional, Tuple, Union, Sequence

# local
import ivy


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):

    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding == 'VALID':
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == 'SAME':
        dim_size = dim_size * stride_size

    return dim_size


# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
    *,
    out: Optional[torch.Tensor] = None
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
    if padding == "SAME":
        if x_shape % strides == 0:
            pad_w = max(f_w_after_dilation - strides, 0)
        else:
            pad_w = max(f_w_after_dilation - (x_shape % strides), 0)
        x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2], value=0)
    elif padding != "VALID":
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv1d(x, filters, None, strides, "valid", dilations)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x,
    filters,
    strides: int,
    padding: str,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NWC",
    dilations: int = 1,
    *,
    out: Optional[torch.Tensor] = None
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(1, 2, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if padding == "VALID":
        padding_list: List[int] = [0]
    elif padding == "SAME":
        padding_list: List[int] = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose1d(
        x, filters, None, strides, padding_list, dilation=dilations
    )
    return res.permute(0, 2, 1)


conv1d.unsupported_dtypes = ("float16",)


# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    *,
    out: Optional[torch.Tensor] = None
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

    if padding == "SAME":
        if x_shape[1] % strides[1] == 0:
            pad_w = max(filter_shape[1] - strides[1], 0)
        else:
            pad_w = max(filter_shape[1] - (x_shape[1] % strides[1]), 0)

        if x_shape[0] % strides[0] == 0:
            pad_h = max(filter_shape[0] - strides[0], 0)
        else:
            pad_h = max(filter_shape[0] - (x_shape[0] % strides[0]), 0)
        x = torch.nn.functional.pad(
            x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
        )
    elif padding != "VALID":
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
    output_shape=None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]]= 1,
    *,
    out: Optional[torch.Tensor] = None
):
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filter_shape = list(filters.shape[0:2])
    filters = filters.permute(3, 2, 0, 1)

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    new_h = _deconv_length(
        x.shape[2],
        strides[0],
        filter_shape[0],
        padding,
        dilations[0]
    )
    new_w = _deconv_length(
        x.shape[3],
        strides[1],
        filter_shape[1],
        padding,
        dilations[1]
    )
    output_shape = [new_h, new_w]
    not_valid_h = False
    not_valid_w = False
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
        out_h = (x.shape[2] - 1) * strides[0] + dilations[0] * (filters.shape[2] - 1) + 1
        out_w = (x.shape[3] - 1) * strides[1] + dilations[1] * (filters.shape[3] - 1) + 1
        output_padding = [max(new_h - out_h, 0), max(new_w - out_w, 0)]
    elif padding == "SAME":
        filter_shape[0] = filter_shape[0] + (filter_shape[0] - 1) * (dilations[0] - 1)
        filter_shape[1] = filter_shape[1] + (filter_shape[1] - 1) * (dilations[1] - 1)
        if output_shape[1] % strides[1] == 0:
            pad_w = max(filter_shape[1] - strides[1], 0)
        else:
            pad_w = max(filter_shape[1] - (output_shape[1] % strides[1]), 0)

        if output_shape[0] % strides[0] == 0:
            pad_h = max(filter_shape[0] - strides[0], 0)
        else:
            pad_h = max(filter_shape[0] - (output_shape[0] % strides[0]), 0)

        if pad_h % 2 != 0:
            pad_h -= 1
            not_valid_h = True
        if pad_w % 2 != 0:
            pad_w -= 1
            not_valid_w = True
        pad_h_ = pad_h // 2
        pad_w_ = pad_w // 2
        out_h = (x.shape[2] - 1) * strides[0] - 2 * pad_h_ + dilations[0] * \
                (filters.shape[2] - 1) + 1
        out_w = (x.shape[3] - 1) * strides[1] - 2 * pad_w_ + dilations[1] * \
                (filters.shape[3] - 1) + 1
        padding_list = [pad_h_, pad_w_]
        output_padding = [max(new_h - out_h, 0), max(new_w - out_w, 0)]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose2d(
        x,
        filters,
        None,
        strides,
        padding_list,
        dilation=dilations,
        output_padding=output_padding
    )
    if not_valid_h:
        res = res[:, :, 0:-1, :]
    if not_valid_w:
        res = res[:, :, :, 0:-1]
    if data_format == "NHWC":
        res = res.permute(0, 2, 3, 1)
    return res


conv2d_transpose.unsupported_dtypes = ('float16',)


# noinspection PyUnresolvedReferences
def depthwise_conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations

    filter_shape = list(filters.shape[0:2])
    dims_in = filters.shape[-1]
    filters = torch.unsqueeze(filters, -1)
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    if padding == "SAME":
        if x_shape[1] % strides[1] == 0:
            pad_w = max(filter_shape[1] - strides[1], 0)
        else:
            pad_w = max(filter_shape[1] - (x_shape[1] % strides[1]), 0)

        if x_shape[0] % strides[0] == 0:
            pad_h = max(filter_shape[0] - strides[0], 0)
        else:
            pad_h = max(filter_shape[0] - (x_shape[0] % strides[0]), 0)
        x = torch.nn.functional.pad(
            x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0
        )

    elif padding != "VALID":
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


depthwise_conv2d.unsupported_dtypes = ('float16', )


# noinspection PyUnresolvedReferences
def conv3d(
    x,
    filters,
    strides: int,
    padding: str,
    data_format: str = "NDHWC",
    dilations: int = 1,
    *,
    out: Optional[torch.Tensor] = None
):
    filter_shape = list(filters.shape[0:3])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if padding == "VALID":
        padding_list: List[int] = [0, 0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv3d(x, filters, None, strides, padding_list, dilations)
    return res.permute(0, 2, 3, 4, 1)


conv3d.unsupported_dtypes = ("float16",)


# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if padding == "VALID":
        padding_list: List[int] = [0, 0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose3d(
        x, filters, None, strides, padding_list, dilation=dilations
    )
    return res.permute(0, 2, 3, 4, 1)
