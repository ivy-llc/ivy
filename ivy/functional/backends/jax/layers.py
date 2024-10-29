"""Collection of Jax network layers, wrapped to fit Ivy syntax and
signature."""

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


def _get_new_padding_before_conv(
    x,
    filters,
    strides,
    padding,
    dims,
    data_format,
    filter_format,
    dilations,
    x_dilations,
):
    if len(x_dilations) != x_dilations.count(1):
        new_pad = [0] * dims
        x_shape = (
            list(x.shape[1 : dims + 1])
            if data_format == ("NWC" or "NHWC" or "NDHWC")
            else list(x.shape[2:])
        )
        x_shape = [
            x_shape[i] + (x_shape[i] - 1) * (x_dilations[i] - 1) for i in range(dims)
        ]
        f_shape = (
            list(filters.shape[:dims])
            if filter_format == "channel_last"
            else list(filters.shape[2:])
        )
        f_shape = [
            f_shape[i] + (f_shape[i] - 1) * (dilations[i] - 1) for i in range(dims)
        ]
        if isinstance(padding, str):
            for i in range(dims):
                new_pad[i] = _handle_padding(
                    x_shape[i], strides[i], f_shape[i], padding
                )
            padding = [
                (new_pad[i] // 2, new_pad[i] - new_pad[i] // 2) for i in range(dims)
            ]
        return padding
    return padding


def conv1d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    data_format = "channel_last" if data_format == "NWC" else "channel_first"
    return conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        dims=1,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
    )


def conv1d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    if data_format == "NWC":
        x_shape = list(x.shape[1:2])
    else:
        x_shape = list(x.shape[2:])
    if filter_format == "channel_first":
        filters = jnp.transpose(filters, (2, 1, 0))
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 1, dilations, output_shape
    )
    res = jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "WIO", data_format),
        True,
    )
    if bias is not None:
        if data_format == "NWC":
            return jnp.add(res, bias)
        return jnp.add(res, bias[(None,) + (...,) + (None,) * 1])
    return res


def conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    data_format = "channel_last" if data_format == "NHWC" else "channel_first"
    return conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        dims=2,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
    )


def conv2d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if data_format == "NHWC":
        x_shape = list(x.shape[1:3])
    else:
        x_shape = list(x.shape[2:])
    if filter_format == "channel_first":
        filters = jnp.transpose(filters, (2, 3, 1, 0))
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 2, dilations, output_shape
    )

    res = jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "HWIO", data_format),
        True,
    )
    if bias is not None:
        if data_format == "NHWC":
            return jnp.add(res, bias)
        return jnp.add(res, bias[(None,) + (...,) + (None,) * 2])
    return res


def depthwise_conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    strides = [strides[1], strides[2]] if len(strides) == 4 else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    if isinstance(padding, int):
        padding = [(padding, padding)] * 2
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
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    data_format = "channel_last" if data_format == "NDHWC" else "channel_first"
    return conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        dims=3,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
    )


def conv3d_transpose(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    filter_format: str = "channel_last",
    data_format: str = "NDHWC",
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    strides = [strides] * 3 if isinstance(strides, int) else strides
    dilations = [dilations] * 3 if isinstance(dilations, int) else dilations
    if filter_format == "channel_first":
        filters = jnp.transpose(filters, (2, 3, 4, 1, 0))
    if data_format == "NDHWC":
        x_shape = list(x.shape[1:4])
    else:
        x_shape = list(x.shape[2:])
    padding = _get_tranpose_padding(
        x_shape, filters.shape, strides, padding, 3, dilations, output_shape
    )
    res = jlax.conv_transpose(
        x,
        filters,
        strides,
        padding,
        dilations,
        (data_format, "DHWIO", data_format),
        True,
    )
    if bias is not None:
        if data_format == "NDHWC":
            return jnp.add(res, bias)
        return jnp.add(res, bias[(None,) + (...,) + (None,) * 3])
    return res


def _get_filter_dataformat(dims: int = 2, filter_format: str = "channel_last"):
    first = True if filter_format == "channel_first" else False
    if dims == 1:
        return "OIW" if first else "WIO"
    if dims == 2:
        return "OIHW" if first else "HWIO"
    elif dims == 3:
        return "OIDHW" if first else "DHWIO"


def conv_general_dilated(
    x: JaxArray,
    filters: JaxArray,
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
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    if isinstance(padding, int):
        padding = [(padding, padding)] * dims
    filter_df = _get_filter_dataformat(dims, filter_format)
    if len(x_dilations) != x_dilations.count(1):
        new_pad = [0] * dims
        x_shape = (
            list(x.shape[1 : dims + 1])
            if data_format == "channel_last"
            else list(x.shape[2:])
        )
        x_shape = [
            x_shape[i] + (x_shape[i] - 1) * (x_dilations[i] - 1) for i in range(dims)
        ]
        f_shape = (
            list(filters.shape[:dims])
            if filter_format == "channel_last"
            else list(filters.shape[2:])
        )
        f_shape = [
            f_shape[i] + (f_shape[i] - 1) * (dilations[i] - 1) for i in range(dims)
        ]
        if isinstance(padding, str):
            for i in range(dims):
                new_pad[i] = _handle_padding(
                    x_shape[i], strides[i], f_shape[i], padding
                )
            padding = [
                (new_pad[i] // 2, new_pad[i] - new_pad[i] // 2) for i in range(dims)
            ]
    df = _get_x_data_format(dims, data_format)
    promoted_type = jnp.promote_types(x.dtype, filters.dtype)
    x = jnp.astype(x, promoted_type)
    filters = jnp.astype(filters, promoted_type)
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
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    filter_format: str = "channel_last",
    data_format: str = "channel_last",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[JaxArray] = None,
    out: Optional[JaxArray] = None,
):
    strides = [strides] * dims if isinstance(strides, int) else strides
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    if filter_format == "channel_first":
        filters = jnp.transpose(filters, (*range(2, dims + 2), 1, 0))
    df = _get_x_data_format(dims, "channel_last")
    filter_df = _get_filter_dataformat(dims)
    if data_format == "channel_first":
        x = jnp.transpose(x, (0, *range(2, dims + 2), 1))
    padding = _get_tranpose_padding(
        x.shape[1:], filters.shape, strides, padding, dims, dilations, output_shape
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


def nms(
    boxes,
    scores=None,
    iou_threshold=0.5,
    max_output_size=None,
    score_threshold=float("-inf"),
):
    change_id = False
    if score_threshold != float("-inf") and scores is not None:
        keep_idx = scores > score_threshold
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        change_id = True
        nonzero = jnp.nonzero(keep_idx)[0].flatten()
    if scores is None:
        scores = jnp.ones((boxes.shape[0],), dtype=boxes.dtype)

    if len(boxes) < 2:
        if len(boxes) == 1:
            ret = jnp.array([0], dtype=ivy.int64)
        else:
            ret = jnp.array([], dtype=ivy.int64)
    else:
        areas = jnp.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)
        order = jnp.argsort(-1 * scores)  # get boxes with more ious first
        boxes = boxes[order]
        areas = areas[order]
        size = order.size
        pad_width = 1 if size == 0 else 2 ** (size - 1).bit_length()

        order = jnp.pad(order, [0, pad_width - size], constant_values=pad_width)
        boxes = jnp.pad(boxes, [[0, pad_width - size], [0, 0]])
        areas = jnp.pad(areas, [0, pad_width - size])
        keep = jnp.zeros((size,), dtype=jnp.int64)
        keep_idx = 0

        def body_fn(loop_vars):
            keep, keep_idx, boxes, areas, order = loop_vars
            max_iou_idx = order[0]
            keep = keep.at[keep_idx].set(max_iou_idx)
            keep_idx += 1
            boxes1 = jnp.maximum(boxes[0, :2], boxes[1:, :2])
            boxes2 = jnp.minimum(boxes[0, 2:4], boxes[1:, 2:4])
            boxes_intersection = jnp.maximum(0.0, boxes2 - boxes1)
            intersection = jnp.prod(
                jnp.where(boxes_intersection != 0, boxes_intersection, 1), axis=1
            )
            iou = intersection / (areas[0] + areas[1:] - intersection)
            condition = jnp.pad(iou <= iou_threshold, [1, 0], constant_values=False)
            order = jnp.where(condition, order, pad_width)
            boxes = jnp.where(jnp.expand_dims(condition, axis=1), boxes, 0)
            areas = jnp.where(condition, areas, 0)
            first = jnp.argwhere(order < pad_width, size=pad_width)[0][0]
            forward = jnp.array([0, first])
            order = order.at[forward].set(order[forward[::-1]])
            boxes = boxes.at[forward].set(boxes[forward[::-1]])
            areas = areas.at[forward].set(areas[forward[::-1]])

            return keep, keep_idx, boxes, areas, order

        def cond_fn(loop_vars):
            _, _, _, _, order = loop_vars
            return jnp.min(order) != jnp.max(order)

        init_vars = (keep, keep_idx, boxes, areas, order)
        keep, keep_idx, boxes, _, _ = jlax.while_loop(cond_fn, body_fn, init_vars)

        ret = jnp.array(keep[:keep_idx], dtype=jnp.int64)

    if len(ret) > 1 and scores is not None:
        ret = sorted(
            ret.flatten().tolist(), reverse=True, key=lambda x: (scores[x], -x)
        )
        ret = jnp.array(ret, dtype=jnp.int64).flatten()

    if change_id and len(ret) > 0:
        ret = jnp.array(nonzero[ret], dtype=jnp.int64).flatten()

    return ret.flatten()[:max_output_size]
