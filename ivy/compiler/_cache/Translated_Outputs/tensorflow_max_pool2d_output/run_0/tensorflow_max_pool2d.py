import tensorflow

from typing import Union
from typing import Optional
from typing import List
from typing import Tuple

from .tensorflow__helpers import tensorflow__determine_depth_max_pooling
from .tensorflow__helpers import tensorflow__handle_padding_bknd
from .tensorflow__helpers import tensorflow__padding_ceil_mode_bknd
from .tensorflow__helpers import tensorflow__validate_max_pool_params_bknd
from .tensorflow__helpers import tensorflow_dev


def tensorflow_max_pool2d(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    dims = 2
    kernel, strides, padding, dilation = tensorflow__validate_max_pool_params_bknd(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )
    permuted_x = False
    if data_format == "NCHW" and tensorflow_dev(x) == "cpu":
        x = tensorflow.transpose(x, (0, 2, 3, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 1]] if len(kernel) == dims + 2 else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 1]] if len(strides) == dims + 2 else strides
        )
        data_format = "NHWC"
        permuted_x = True
    x, kernel, strides, depth_pooling = tensorflow__determine_depth_max_pooling(
        x, kernel, strides, dims, data_format=data_format
    )
    if not depth_pooling:
        if ceil_mode:
            new_kernel = [
                (kernel[i] + (kernel[i] - 1) * (dilation[i] - 1)) for i in range(dims)
            ]
            if data_format == "NCHW":
                x_shape = x.shape[2:]
            else:
                x_shape = x.shape[1:-1]
            if isinstance(padding, str):
                pad_h = tensorflow__handle_padding_bknd(
                    x_shape[0], strides[0], new_kernel[0], padding
                )
                pad_w = tensorflow__handle_padding_bknd(
                    x_shape[1], strides[1], new_kernel[1], padding
                )
                padding = [
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ]
            for i in range(dims):
                padding[i] = tensorflow__padding_ceil_mode_bknd(
                    x_shape[i], new_kernel[i], padding[i], strides[i]
                )
        if isinstance(padding, list):
            if any(item != 0 for sublist in padding for item in sublist):
                if len(padding) < dims + 2:
                    if data_format == "NCHW":
                        padding = [(0, 0), (0, 0), *padding]
                    else:
                        padding = [(0, 0), *padding, (0, 0)]
                x = tensorflow.pad(
                    x, padding, constant_values=tensorflow.math.reduce_min(x)
                )
            padding = "VALID"
    elif isinstance(padding, list):
        if any(item != 0 for sublist in padding for item in sublist):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        else:
            padding = "VALID"
    if any(d > 1 for d in dilation):
        res = tensorflow.nn.pool(
            x,
            kernel,
            "MAX",
            strides,
            padding,
            dilations=dilation,
            data_format=data_format,
        )
    else:
        res = tensorflow.nn.max_pool2d(
            x, kernel, strides, padding, data_format=data_format
        )
    if depth_pooling:
        res = tensorflow.transpose(res, (0, 2, 3, 1))
    if permuted_x:
        return tensorflow.transpose(res, (0, 3, 1, 2))
    return res
