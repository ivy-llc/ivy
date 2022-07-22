"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

# global
import math
import torch
from typing import List, Optional, Tuple, Union, Sequence

# local
import ivy


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


conv1d.unsupported_dtypes = ('float16',)


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


conv2d.unsupported_dtypes = ('float16',)


# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: int,
    padding: str,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: str = "NHWC",
    dilations: int = 1,
    *,
    out: Optional[torch.Tensor] = None
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose2d(
        x, filters, None, strides, padding_list, dilation=dilations
    )
    return res.permute(0, 2, 3, 1)


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
    filter_shape = list(filters.shape[0:2])
    dims_in = filters.shape[-1]
    filters = torch.unsqueeze(filters, -1)
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    # noinspection PyArgumentEqualDefault
    res = torch.nn.functional.conv2d(
        x, filters, None, strides, padding_list, dilations, dims_in
    )
    return res.permute(0, 2, 3, 1)


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


conv3d.unsupported_dtypes = ('float16',)


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
