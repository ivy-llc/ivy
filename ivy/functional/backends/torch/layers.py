"""Collection of PyTorch network layers, wrapped to fit Ivy syntax and signature."""

# global
import math as _math
import torch
from typing import List, Optional, Tuple, Union


def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor, 
    scale: float, 
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    attention = torch.matmul(q, k.transpose(2, 3)) * scale

    if mask is not None:
        attention = attention.masked_fill(mask == 0, -1e9) 

    attention = torch.nn.functional.softmax(attention, dim=-1)
    output = torch.matmul(attention, v)

    return output, attention

# noinspection PyUnresolvedReferences
def conv1d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
) -> torch.Tensor:
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(2, 1, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if padding == "VALID":
        padding_list: List[int] = [0]
    elif padding == "SAME":
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv1d(x, filters, None, strides, padding_list, dilations)
    return res.permute(0, 2, 1)


# noinspection PyUnresolvedReferences
def conv1d_transpose(
    x,
    filters,
    strides: int,
    padding: str,
    output_shape: Optional[List[int]] = None,
    data_format: str = "NWC",
    dilations: int = 1,
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(1, 2, 0)
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if padding == "VALID":
        padding_list: List[int] = [0]
    elif padding == "SAME":
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose1d(
        x, filters, None, strides, padding_list, dilation=dilations
    )
    return res.permute(0, 2, 1)


# noinspection PyUnresolvedReferences
def conv2d(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: int = 1,
) -> torch.Tensor:
    filter_shape = list(filters.shape[0:2])
    filters = filters.permute(3, 2, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [int(_math.floor(item / 2)) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv2d(x, filters, None, strides, padding_list, dilations)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


# noinspection PyUnresolvedReferences
def conv2d_transpose(
    x: torch.Tensor,
    filters: torch.Tensor,
    strides: int,
    padding: str,
    output_shape: Optional[List[int]] = None,
    data_format: str = "NHWC",
    dilations: int = 1,
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(2, 3, 0, 1)
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if padding == "VALID":
        padding_list: List[int] = [0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
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
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
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
):
    filter_shape = list(filters.shape[0:3])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if padding == "VALID":
        padding_list: List[int] = [0, 0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv3d(x, filters, None, strides, padding_list, dilations)
    return res.permute(0, 2, 3, 4, 1)


# noinspection PyUnresolvedReferences
def conv3d_transpose(
    x,
    filters,
    strides: int,
    padding: str,
    output_shape: Optional[List[int]] = None,
    data_format: str = "NDHWC",
    dilations: int = 1,
):
    filter_shape = list(filters.shape[0:1])
    filters = filters.permute(3, 4, 0, 1, 2)
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 3)
    if padding == "VALID":
        padding_list: List[int] = [0, 0, 0]
    elif padding == "SAME":
        padding_list: List[int] = [_math.floor(item / 2) for item in filter_shape]
    else:
        raise Exception(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.conv_transpose3d(
        x, filters, None, strides, padding_list, dilation=dilations
    )
    return res.permute(0, 2, 3, 4, 1)
