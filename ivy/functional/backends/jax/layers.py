"""Collection of Jax network layers, wrapped to fit Ivy syntax and signature."""

# global
import jax.lax as jlax
import jax.numpy as jnp

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from typing import Union, Tuple, Optional, Sequence
from ivy.functional.ivy.layers import (
    _handle_padding,
    _deconv_length,
    _get_x_data_format,
)


def _transpose_padding_helper(k, s, padding, dilation, diff=0):
    k = (k - 1) * dilation + 1
    if padding == "SAME":
        pad_len = k + s - 2
        pad_len -= diff
        if s > k - 1:
            pad_a = k - 1
        else:
            pad_a = int(jnp.ceil(pad_len / 2))
    else:
        pad_len = k + s - 2 + max(k - s, 0)
        pad_a = k - 1
    pad_b = pad_len - pad_a
    return pad_a, pad_b


def _get_tranpose_padding(
    x_shape, filter_shape, strides, padding, dims, dilations, output_shape
):
    new_shape = [
        _deconv_length(x_shape[i], strides[i], filter_shape[i], padding, dilations[i])
        for i in range(dims)
    ]
    if output_shape is None:
        output_shape = [x_shape[0], *new_shape, filter_shape[-1]]
    elif len(output_shape) == dims:
        output_shape = [x_shape[0]] + list(output_shape) + [filter_shape[-1]]
    shape_diff = [-(output_shape[1 + i] - new_shape[i]) for i in range(dims)]
    pad_list = [
        _transpose_padding_helper(
            filter_shape[i], strides[i], padding, dilations[i], shape_diff[i]
        )
        for i in range(dims)
    ]
    return pad_list


def conv1d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x, filters, strides, padding, None, dilations, (data_format, "WIO", data_format)
    )


def conv1d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: int,
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "NWC",
    dilations: Optional[int] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NWC":
        x_shape = list(x.shape[1:2])
    else:
        x_shape = list(x.shape[2:])
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 1, dilations, output_shape
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "WIO", data_format),
        True,
    )


def conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "HWIO", data_format),
    )


def conv2d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NHWC":
        x_shape = list(x.shape[1:3])
    else:
        x_shape = list(x.shape[2:])
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 2, dilations, output_shape
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "HWIO", data_format),
        True,
    )


def depthwise_conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NHWC",
    dilations: Optional[Union[int, Tuple[int, int]]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    strides = [strides[1], strides[2]] if len(strides) == 4 else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    filters = jnp.squeeze(filters, 3) if filters.ndim == 4 else filters
    cn = filters.shape[-1]
    filters = jnp.expand_dims(filters, -2)
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "HWIO", data_format),
        feature_group_count=cn,
    )


def conv3d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: Optional[str] = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "DHWIO", data_format),
    )


def conv3d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    data_format: Optional[str] = "NDHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    if data_format == "NDHWC":
        x_shape = list(x.shape[1:4])
    else:
        x_shape = list(x.shape[2:])
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 3, dilations, output_shape
    )
    return jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "DHWIO", data_format),
        True,
    )


def _get_filter_dataformat(dims: int = 2):
    if dims == 1:
        return "WIO"
    if dims == 2:
        return "HWIO"
    elif dims == 3:
        return "DHWIO"


def conv_general_dilated(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: Optional[int] = 2,
    data_format: Optional[str] = "channel_last",
    feature_group_count: Optional[int] = 1,
    x_dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    filter_df = _get_filter_dataformat(dims)
    if not len(x_dilations) == x_dilations.count(1):
        new_pad = [0] * dims
        if data_format == "channel_last":
            x_shape = list(x.shape[1 : dims + 1])
        else:
            x_shape = list(x.shape[2:])
        x_shape = [
            x_shape[i] + (x_shape[i] - 1) * (x_dilations[i] - 1) for i in range(dims)
        ]
        filter_shape = list(filters.shape[:dims])
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        if isinstance(padding, str):
            for i in range(dims):
                new_pad[i] = _handle_padding(
                    x_shape[i], strides[i], filter_shape[i], padding
                )
            padding = [
                (new_pad[i] // 2, new_pad[i] - new_pad[i] // 2) for i in range(dims)
            ]
    df = _get_x_data_format(dims, data_format)
    res = jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        x_dilations,
        dilations,
        (df, filter_df, df),
        feature_group_count,
    )
    if bias is not None:
        if data_format == "channel_last":
            return jnp.add(res, bias)
        return jnp.add(res, bias[(None,) + (...,) + (None,) * dims])
    return res


def conv_general_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: Optional[int] = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    data_format: Optional[str] = "channel_last",
    dilations: Optional[
        Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
    ] = 1,
    feature_group_count: Optional[int] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    filters = jnp.swapaxes(filters, -1, -2)
    df = _get_x_data_format(dims, "channel_last")
    filter_df = _get_filter_dataformat(dims)
    if data_format == "channel_first":
        x = jnp.transpose(x, (0, *range(2, dims + 2), 1))
    x_shape = list(x.shape[1 : dims + 1])
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, dims, dilations, output_shape
    )
    res = jnp.concatenate(
        [
            jlax.conv_transpose(
                x[..., j : j + filters.shape[-1] // feature_group_count],
                filters[..., j : j + filters.shape[-1] // feature_group_count],
                strides,
                padding,
                dilations,
                (df, filter_df, df),
                True,
            )
            for j in range(
                0, filters.shape[-1], filters.shape[-1] // feature_group_count
            )
        ],
        axis=-1,
    )
    res = jnp.add(res, bias) if bias is not None else res
    if data_format == "channel_first":
        return jnp.transpose(res, (0, dims + 1, *range(1, dims + 1)))
    return res
