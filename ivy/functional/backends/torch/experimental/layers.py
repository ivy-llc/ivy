# global
from typing import Optional, Union, Tuple
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def max_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(strides, int):
        strides = (strides, strides)
    elif len(strides) == 1:
        strides = (strides[0], strides[0])

    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = torch.nn.functional.pad(
        x,
        [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
        value=float("-inf"),
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.max_pool2d(x, kernel, strides, 0)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def max_pool1d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
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

    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    x_shape = x.shape[2]
    pad_w = ivy.handle_padding(x_shape, strides[0], kernel[0], padding)
    x = torch.nn.functional.pad(
        x, [pad_w // 2, pad_w - pad_w // 2], value=float("-inf")
    )

    res = torch.nn.functional.max_pool1d(x, kernel, strides, 0)

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
def max_pool3d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
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
    x_shape = list(x.shape[2:])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], kernel[2], padding)
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
        value=float("-inf"),
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.max_pool3d(x, kernel, strides, 0)
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
def avg_pool3d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
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
    x_shape = list(x.shape[2:])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], kernel[2], padding)
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
        mode="replicate",
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.avg_pool3d(x, kernel, strides, 0)
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
def avg_pool2d(
    x: torch.Tensor,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
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
    x_shape = list(x.shape[2:])
    pad_h = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_w = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    x = torch.nn.functional.pad(
        x,
        [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
        mode="replicate",
    )
    if padding != "VALID" and padding != "SAME":
        raise ivy.exceptions.IvyException(
            "Invalid padding arg {}\n"
            'Must be one of: "VALID" or "SAME"'.format(padding)
        )
    res = torch.nn.functional.avg_pool2d(x, kernel, strides, 0)
    if data_format == "NHWC":
        return res.permute(0, 2, 3, 1)
    return res
