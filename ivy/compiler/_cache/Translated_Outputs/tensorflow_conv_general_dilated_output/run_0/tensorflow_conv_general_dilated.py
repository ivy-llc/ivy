import tensorflow

from typing import Union
from typing import Sequence
from typing import Tuple
from typing import Optional

from .tensorflow__helpers import tensorflow__extend_2d_padding
from .tensorflow__helpers import tensorflow__extend_3d_strides_dilations
from .tensorflow__helpers import tensorflow__get_x_data_format_bknd
from .tensorflow__helpers import tensorflow__pad_before_conv
from .tensorflow__helpers import tensorflow__x_dil_before_conv
from .tensorflow__helpers import tensorflow_depthwise_conv2d
from .tensorflow__helpers import tensorflow_dev
from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_conv_general_dilated(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    filters: Union[tensorflow.Tensor, tensorflow.Variable],
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
    bias: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if filter_format == "channel_first":
        filters = tensorflow.transpose(filters, (*range(2, dims + 2), 1, 0))
    num_channels = x.shape[1] if data_format == "channel_first" else x.shape[-1]
    if filters.shape[-2] != num_channels // feature_group_count:
        raise Exception(
            f"given feature_group_count {feature_group_count} expected input channel of the filter to be {num_channels // feature_group_count} but got {filters.shape[-2]}"
        )
    if num_channels % feature_group_count != 0:
        raise Exception(
            f"input channel should be divisible by feature group count {feature_group_count} but got input channel {num_channels}"
        )
    permuted_x = False
    if data_format == "channel_first" and (
        tensorflow_dev(x) == "cpu" or feature_group_count != 1
    ):
        x = tensorflow.transpose(x, (0, *range(2, dims + 2), 1))
        data_format = "channel_last"
        permuted_x = True
    data_format = tensorflow__get_x_data_format_bknd(dims, data_format)
    x = tensorflow__x_dil_before_conv(x, dims, x_dilations, data_format)
    if dims == 2:
        padding = tensorflow__extend_2d_padding(padding, data_format)
        if feature_group_count == 1:
            res = tensorflow.nn.conv2d(
                x,
                filters,
                strides,
                padding,
                data_format=data_format,
                dilations=dilations,
            )
        else:
            if not isinstance(padding, str):
                padding = padding[1:-1]
            res = tensorflow_depthwise_conv2d(
                x,
                tensorflow.transpose(filters, (0, 1, 3, 2)),
                strides,
                padding,
                data_format=data_format,
                dilations=dilations,
            )
    else:
        x, padding = tensorflow__pad_before_conv(x, padding, dims, data_format)
        if dims == 1:
            if feature_group_count == 1:
                res = tensorflow.nn.conv1d(
                    x,
                    filters,
                    strides,
                    padding,
                    data_format=data_format,
                    dilations=dilations,
                )
            else:
                res = tensorflow.concat(
                    [
                        tensorflow.nn.conv1d(
                            x[..., i : i + filters.shape[-2]],
                            filters[
                                ..., j : j + filters.shape[-1] // feature_group_count
                            ],
                            strides,
                            padding,
                            data_format,
                            dilations,
                        )
                        for i, j in zip(
                            range(0, x.shape[-1], filters.shape[-2]),
                            range(
                                0,
                                filters.shape[-1],
                                filters.shape[-1] // feature_group_count,
                            ),
                        )
                    ],
                    axis=-1,
                )
        else:
            strides, dilations = tensorflow__extend_3d_strides_dilations(
                strides, dilations, data_format
            )
            if feature_group_count == 1:
                res = tensorflow.nn.conv3d(
                    x,
                    filters,
                    strides,
                    padding,
                    data_format=data_format,
                    dilations=dilations,
                )
            else:
                res = tensorflow.concat(
                    [
                        tensorflow.nn.conv3d(
                            x[..., i : i + filters.shape[-2]],
                            filters[
                                ..., j : j + filters.shape[-1] // feature_group_count
                            ],
                            strides,
                            padding,
                            data_format,
                            dilations,
                        )
                        for i, j in zip(
                            range(0, x.shape[-1], filters.shape[-2]),
                            range(
                                0,
                                filters.shape[-1],
                                filters.shape[-1] // feature_group_count,
                            ),
                        )
                    ],
                    axis=-1,
                )
    if bias is not None:
        if data_format[1] == "C":
            bias = tensorflow.reshape(bias, [1, -1, *([1] * dims)])
        res = tensorflow.math.add(res, bias)
    if permuted_x:
        return tensorflow.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res
