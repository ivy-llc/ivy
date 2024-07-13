# global
import math
from typing import Union, Optional, Tuple, List, Literal, Sequence, Callable
import tensorflow as tf

# local
from ivy.func_wrapper import (
    inputs_to_ivy_arrays,
    output_to_native_arrays,
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from .. import backend_version
import ivy
from ivy.functional.ivy.layers import (
    _handle_padding,
    _get_num_padded_values,
    _validate_max_pool_params,
    _depth_max_pooling_helper,
)
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode, _get_size


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_last"):
    # Determine depth pooling
    kernel, strides, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        x = tf.transpose(x, (0, dims + 1, *range(1, dims + 1)))
    return x, kernel, strides, depth_pooling


def max_pool1d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dims = 1
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    permuted_x = False
    if data_format == "NCW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 1))
        kernel = [kernel[i] for i in [0, 2, 1]] if len(kernel) == (dims + 2) else kernel
        strides = (
            [strides[i] for i in [0, 2, 1]] if len(strides) == (dims + 2) else strides
        )
        data_format = "NWC"
        permuted_x = True

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format=data_format
    )

    if not depth_pooling:
        if ceil_mode:
            new_kernel = [kernel[0] + (kernel[0] - 1) * (dilation[0] - 1)]
            if data_format == "NCW":
                x_shape = x.shape[2:]
            else:
                x_shape = x.shape[1:-1]
            if isinstance(padding, str):
                pad_w = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
                padding = [(pad_w // 2, pad_w - pad_w // 2)]
            padding[0] = _padding_ceil_mode(
                x_shape[0], new_kernel[0], padding[0], strides[0]
            )
        if isinstance(padding, list):
            if any(item != 0 for sublist in padding for item in sublist):
                if len(padding) < dims + 2:
                    if data_format == "NCW":
                        padding = [(0, 0), (0, 0), *padding]
                    else:
                        padding = [(0, 0), *padding, (0, 0)]
                x = tf.pad(x, padding, constant_values=tf.math.reduce_min(x))
            padding = "VALID"
    elif isinstance(padding, list):
        if any(item != 0 for sublist in padding for item in sublist):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        else:
            padding = "VALID"
    res = tf.nn.pool(
        x, kernel, "MAX", strides, padding, dilations=dilation, data_format=data_format
    )

    if depth_pooling:
        res = tf.transpose(res, (0, 2, 1))
    if permuted_x:
        return tf.transpose(res, (0, 2, 1))
    return res


def max_pool2d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dims = 2
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    permuted_x = False
    if data_format == "NCHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 1]] if len(kernel) == (dims + 2) else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 1]]
            if len(strides) == (dims + 2)
            else strides
        )
        data_format = "NHWC"
        permuted_x = True

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format=data_format
    )

    if not depth_pooling:
        if ceil_mode:
            new_kernel = [
                kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(dims)
            ]
            if data_format == "NCHW":
                x_shape = x.shape[2:]
            else:
                x_shape = x.shape[1:-1]
            if isinstance(padding, str):
                pad_h = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
                pad_w = _handle_padding(x_shape[1], strides[1], new_kernel[1], padding)
                padding = [
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ]
            for i in range(dims):
                padding[i] = _padding_ceil_mode(
                    x_shape[i], new_kernel[i], padding[i], strides[i]
                )
        if isinstance(padding, list):
            if any(item != 0 for sublist in padding for item in sublist):
                if len(padding) < dims + 2:
                    if data_format == "NCHW":
                        padding = [(0, 0), (0, 0), *padding]
                    else:
                        padding = [(0, 0), *padding, (0, 0)]
                x = tf.pad(x, padding, constant_values=tf.math.reduce_min(x))
            padding = "VALID"
    elif isinstance(padding, list):
        if any(item != 0 for sublist in padding for item in sublist):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        else:
            padding = "VALID"
    if any(d > 1 for d in dilation):
        res = tf.nn.pool(
            x,
            kernel,
            "MAX",
            strides,
            padding,
            dilations=dilation,
            data_format=data_format,
        )
    else:  # faster
        res = tf.nn.max_pool2d(x, kernel, strides, padding, data_format=data_format)

    if depth_pooling:
        res = tf.transpose(res, (0, 2, 3, 1))
    if permuted_x:
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def max_pool3d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dims = 3
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    permuted_x = False
    if data_format == "NCDHW" and ivy.dev(x) == "cpu":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 4, 1]]
            if len(kernel) == (dims + 2)
            else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 4, 1]]
            if len(strides) == (dims + 2)
            else strides
        )
        data_format = "NDHWC"
        permuted_x = True

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, dims, data_format=data_format
    )

    if not depth_pooling:
        if ceil_mode:
            new_kernel = [
                kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(dims)
            ]
            if data_format == "NCDHW":
                x_shape = x.shape[2:]
            else:
                x_shape = x.shape[1:-1]
            if isinstance(padding, str):
                pad_d = _handle_padding(x_shape[0], strides[0], new_kernel[0], padding)
                pad_h = _handle_padding(x_shape[1], strides[1], new_kernel[1], padding)
                pad_w = _handle_padding(x_shape[2], strides[2], new_kernel[2], padding)
                padding = [
                    (pad_d // 2, pad_d - pad_d // 2),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ]
            for i in range(dims):
                padding[i] = _padding_ceil_mode(
                    x_shape[i], new_kernel[i], padding[i], strides[i]
                )
        if isinstance(padding, list):
            if any(item != 0 for sublist in padding for item in sublist):
                if len(padding) < dims + 2:
                    if data_format == "NCDHW":
                        padding = [(0, 0), (0, 0), *padding]
                    else:
                        padding = [(0, 0), *padding, (0, 0)]
                x = tf.pad(x, padding, constant_values=tf.math.reduce_min(x))
            padding = "VALID"
    elif isinstance(padding, list):
        if any(item != 0 for sublist in padding for item in sublist):
            raise NotImplementedError(
                "Nonzero explicit padding is not supported for depthwise max pooling"
            )
        else:
            padding = "VALID"
    res = tf.nn.pool(
        x, kernel, "MAX", strides, padding, dilations=dilation, data_format=data_format
    )

    if depth_pooling:
        res = tf.transpose(res, (0, 2, 3, 4, 1))
    if permuted_x:
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


def _handle_manual_pad_avg_pool(x, kernel, strides, padding, ceil_mode, dims):
    if isinstance(padding, str):
        pad_specific = [
            _handle_padding(x.shape[i + 1], strides[i], kernel[i], padding)
            for i in range(dims)
        ]
        padding = [
            (pad_specific[i] // 2, pad_specific[i] - pad_specific[i] // 2)
            for i in range(dims)
        ]
    else:
        if isinstance(padding, int):
            padding = [(padding,) * 2] * dims
        pad_specific = [sum(padding[i]) for i in range(dims)]
    c = []
    if ceil_mode:
        for i in range(dims):
            padding[i], c_i = _padding_ceil_mode(
                x.shape[i + 1], kernel[i], padding[i], strides[i], True
            )
            c.append(c_i)
            pad_specific[i] = sum(padding[i])
    return padding, pad_specific, c


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "float64")}, backend_version)
def avg_pool1d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(kernel, int):
        kernel = [kernel]
    elif len(kernel) == 1:
        kernel = [kernel[0]]

    if isinstance(strides, int):
        strides = [strides]
    elif len(strides) == 1:
        strides = [strides[0]]

    if data_format in ("NCW", "NCL"):
        print("why")
        x = tf.transpose(x, (0, 2, 1))

    manual_padding = False
    # Have to manually pad if explicit padding is provided, or if ceil_mode is True
    if not isinstance(padding, str) or ceil_mode or count_include_pad:
        padding, pad_specific, c = _handle_manual_pad_avg_pool(
            x, kernel, strides, padding, ceil_mode, 1
        )
        x = tf.pad(x, [(0, 0), *padding, (0, 0)], constant_values=0)
        manual_padding = True
        padding = "VALID"

    res = tf.nn.avg_pool1d(x, kernel, strides, padding)

    # removing any manual padding added because of ceil_mode or count_include_pad
    if (manual_padding and not count_include_pad) or ceil_mode:
        if not count_include_pad:
            num_padded_values = tf.convert_to_tensor(
                ivy.map(
                    _get_num_padded_values,
                    constant={
                        "p": pad_specific[0],
                        "n": x.shape[1] - pad_specific[0],
                        "k": kernel[0],
                        "s": strides[0],
                    },
                    unique={
                        "i": tf.range(res.shape[1]),
                    },
                ),
                dtype=res.dtype,
            )
        else:
            num_padded_values = tf.scatter_nd(
                tf.constant([[res.shape[1] - 1]]),
                tf.constant([c[0]], dtype=res.dtype),
                tf.constant([res.shape[1]], dtype=tf.int32),
            )
        res = (kernel[0] * res) / (kernel[0] - num_padded_values[:, None])

    if data_format in ("NCW", "NCL"):
        res = tf.transpose(res, (0, 2, 1))
    return res


@with_unsupported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def avg_pool2d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(kernel, int):
        kernel = (kernel,) * 2
    elif len(kernel) == 1:
        kernel = (kernel[0],) * 2

    if isinstance(strides, int):
        strides = (strides,) * 2
    elif len(strides) == 1:
        strides = (strides[0],) * 2

    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    manual_padding = False
    # Have to manually pad if explicit padding is provided, or if ceil_mode is True
    if not isinstance(padding, str) or ceil_mode or count_include_pad:
        padding, pad_specific, c = _handle_manual_pad_avg_pool(
            x, kernel, strides, padding, ceil_mode, 2
        )
        x = tf.pad(x, [(0, 0), *padding, (0, 0)], constant_values=0)
        manual_padding = True
        padding = "VALID"

    if divisor_override is not None:
        # sum pooling then dividing by divisor_override if it is provided
        res = tf.nn.depthwise_conv2d(
            x, tf.ones(kernel + (x.shape[-1], 1)), (1,) + strides + (1,), padding
        )
        res = res / divisor_override
    else:
        res = tf.nn.avg_pool2d(x, kernel, strides, padding)

    # removing any manual padding added because of ceil_mode or count_include_pad
    if (manual_padding and not count_include_pad) or ceil_mode and not divisor_override:
        if not count_include_pad:
            num_padded_values = [
                tf.convert_to_tensor(
                    ivy.map(
                        _get_num_padded_values,
                        constant={
                            "p": pad_specific[i],
                            "n": x.shape[i + 1] - pad_specific[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={
                            "i": tf.range(res.shape[i + 1]),
                        },
                    ),
                    dtype=res.dtype,
                )
                for i in range(2)
            ]
        else:
            num_padded_values = []
            for i in range(2):
                num_pad = tf.scatter_nd(
                    tf.constant([[res.shape[i + 1] - 1]]),
                    tf.constant([c[i]], dtype=res.dtype),
                    tf.constant([res.shape[i + 1]], dtype=tf.int32),
                )
                num_padded_values.append(num_pad)
        num_padded_values1 = num_padded_values[0][:, None]
        num_padded_values2 = num_padded_values[1][None, :]
        num_padded_values = (
            num_padded_values1 * kernel[1]
            + num_padded_values2 * kernel[0]
            - num_padded_values1 * num_padded_values2
        )
        kernel_mul = tf.cast(tf.math.reduce_prod(kernel), res.dtype)
        res = (kernel_mul * res) / (kernel_mul - tf.expand_dims(num_padded_values, -1))

    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def avg_pool3d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(kernel, int):
        kernel = (kernel,) * 3
    elif len(kernel) == 1:
        kernel = (kernel[0],) * 3

    if isinstance(strides, int):
        strides = (strides,) * 3
    elif len(strides) == 1:
        strides = (strides[0],) * 3

    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))

    manual_padding = False
    # Have to manually pad if explicit padding is provided, or if ceil_mode is True
    if not isinstance(padding, str) or ceil_mode or count_include_pad:
        padding, pad_specific, c = _handle_manual_pad_avg_pool(
            x, kernel, strides, padding, ceil_mode, 3
        )
        x = tf.pad(x, [(0, 0), *padding, (0, 0)], constant_values=0)
        manual_padding = True
        padding = "VALID"

    if divisor_override is not None:
        # sum pooling then dividing by divisor_override if it is provided
        res = ivy.conv_general_dilated(
            x,
            tf.ones(kernel + (1, x.shape[-1])),
            list(strides),
            padding,
            dims=3,
            feature_group_count=x.shape[-1],
        )
        res = res / divisor_override
    else:
        res = tf.nn.avg_pool3d(x, kernel, strides, padding)

    # removing any manual padding added because of ceil_mode or count_include_pad
    if (
        (manual_padding and not count_include_pad) or ceil_mode
    ) and not divisor_override:
        if not count_include_pad:
            num_padded_values = [
                tf.convert_to_tensor(
                    ivy.map(
                        _get_num_padded_values,
                        constant={
                            "p": pad_specific[i],
                            "n": x.shape[i + 1] - pad_specific[i],
                            "k": kernel[i],
                            "s": strides[i],
                        },
                        unique={
                            "i": tf.range(res.shape[i + 1]),
                        },
                    ),
                    dtype=res.dtype,
                )
                for i in range(3)
            ]
        else:
            num_padded_values = []
            for i in range(3):
                num_pad = tf.scatter_nd(
                    tf.constant([[res.shape[i + 1] - 1]]),
                    tf.constant([c[i]], dtype=res.dtype),
                    tf.constant([res.shape[i + 1]], dtype=tf.int32),
                )
                num_padded_values.append(num_pad)
        num_padded_values1 = tf.reshape(num_padded_values[0], (-1, 1, 1))
        num_padded_values2 = tf.reshape(num_padded_values[1], (1, -1, 1))
        num_padded_values3 = tf.reshape(num_padded_values[2], (1, 1, -1))
        num_padded_values = (
            num_padded_values1 * kernel[1] * kernel[2]
            + num_padded_values2 * kernel[0] * kernel[2]
            + num_padded_values3 * kernel[0] * kernel[1]
            + num_padded_values1 * num_padded_values2 * num_padded_values3
            - num_padded_values1 * num_padded_values2 * kernel[2]
            - num_padded_values1 * num_padded_values3 * kernel[1]
            - num_padded_values2 * num_padded_values3 * kernel[0]
        )
        kernel_mul = tf.cast(tf.math.reduce_prod(kernel), res.dtype)
        res = (kernel_mul * res) / (kernel_mul - tf.expand_dims(num_padded_values, -1))

    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


@with_unsupported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def pool(
    x: Union[tf.Tensor, tf.Variable],
    window_shape: Union[int, Tuple[int], Tuple[int, int]],
    pool_type: str,
    /,
    *,
    strides: Optional[Union[int, Tuple[int], Tuple[int, int]]] = None,
    padding: str = "VALID",
    data_format: Optional[str] = None,
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = None,
    ceil_mode: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.pool(
        x,
        window_shape,
        pool_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )


@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, backend_version)
def dct(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    # ToDo: Update this once tf.signal.dct supports axis other than -1
    if axis != -1:
        new_dims = list(range(len(x.shape)))
        if axis < 0:
            axis = len(x.shape) + axis
        new_dims[axis], new_dims[-1] = new_dims[-1], axis
        x = tf.transpose(x, new_dims)
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
        dct_out = tf.transpose(dct_out, new_dims)
    else:
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
    return dct_out


def idct(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return dct(x, type=inverse_type, n=n, axis=axis, norm=norm, out=out)


def _fft_norm(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str = "backward",
):
    n = tf.constant(x.shape[dim], dtype=x.dtype)
    if norm == "backward":
        return x
    elif norm == "ortho":
        return x / tf.cast(tf.sqrt(tf.cast(n, tf.float32)), x.dtype)
    elif norm == "forward":
        return x / tf.cast(n, x.dtype)
    else:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


def _ifft_norm(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    *,
    norm: str = "backward",
):
    n = x.shape[dim]
    if norm == "backward":
        return x
    elif norm == "ortho":
        return x * math.sqrt(n)
    elif norm == "forward":
        return x * n
    else:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


@with_supported_dtypes(
    {"2.15.0 and below": ("complex", "float32", "float64")}, backend_version
)
def fft(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # ToDo: Remove conversion from float to complex when casting mode is working
    if x.dtype == "float32":
        x = tf.cast(x, tf.complex64)
    elif x.dtype == "float64":
        x = tf.cast(x, tf.complex128)
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm not in ["backward", "ortho", "forward"]:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    if x.shape[dim] != n:
        s = list(x.shape)
        if s[dim] > n:
            index = [slice(None)] * len(s)
            index[dim] = slice(0, n)
            x = x[tuple(index)]
            del index
        else:
            s[dim] = n - s[dim]
            z = tf.zeros(s, x.dtype)
            x = tf.concat([x, z], dim)
        del s
    operation_name = f"{n} points FFT at dim {dim} with {norm} normalization"
    if dim != -1 or dim != len(x.shape) - 1:
        permute = [i for i in range(len(x.shape))]
        permute[dim], permute[-1] = permute[-1], permute[dim]
        x = tf.transpose(x, permute)
        ret = tf.signal.fft(x, operation_name)
        ret = tf.transpose(ret, permute)
        del permute
    else:
        ret = tf.signal.fft(x, operation_name)
    ret = _fft_norm(ret, dim, norm=norm)
    return ret


@with_unsupported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def dropout(
    x: Union[tf.Tensor, tf.Variable],
    prob: float,
    /,
    *,
    scale: bool = True,
    dtype: tf.DType = None,
    training: bool = True,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence[int]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x = ivy.astype(x, dtype) if dtype and x.dtype != dtype else x
    if prob == 0 or not training:
        return x
    res = tf.nn.dropout(x, prob, noise_shape=noise_shape, seed=seed)
    res = tf.multiply(res, (1.0 - prob)) if not scale else res
    return res


def dropout1d(
    x: Union[tf.Tensor, tf.Variable],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if training:
        is_batched = len(x.shape) == 3
        if data_format == "NCW":
            perm = (0, 2, 1) if is_batched else (1, 0)
            x = tf.transpose(x, perm)
        res = tf.nn.dropout(x, prob)
        if data_format == "NCW":
            res = tf.transpose(res, perm)
    else:
        res = x
    return res


def dropout2d(
    x: Union[tf.Tensor, tf.Variable],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if training:
        is_batched = len(x.shape) == 4
        if data_format == "NCHW":
            perm = (0, 2, 3, 1) if is_batched else (1, 2, 0)
            x = tf.transpose(x, perm)
        res = tf.nn.dropout(x, prob)
        if data_format == "NCHW":
            perm = (0, 3, 1, 2) if is_batched else (2, 0, 1)
            res = tf.transpose(res, perm)
    else:
        res = x
    return res


def dropout3d(
    x: Union[tf.Tensor, tf.Variable],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if training:
        is_batched = len(x.shape) == 5
        if data_format == "NCDHW":
            perm = (0, 2, 3, 4, 1) if is_batched else (1, 2, 3, 0)
            x = tf.transpose(x, perm)
        res = tf.nn.dropout(x, prob)
        if data_format == "NCDHW":
            perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
            res = tf.transpose(res, perm)
    else:
        res = x
    return res


def ifft(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if n is None:
        n = x.shape[dim]
    if n < -len(x.shape):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm not in ["backward", "ortho", "forward"]:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    if x.shape[dim] != n:
        s = list(x.shape)
        if s[dim] > n:
            index = [slice(None)] * len(s)
            index[dim] = slice(0, n)
            x = x[tuple(index)]
            del index
        else:
            s[dim] = n - s[dim]
            z = tf.zeros(s, x.dtype)
            x = tf.concat([x, z], axis=dim)
        del s
    operation_name = f"{n} points FFT at dim {dim} with {norm} normalization"
    if dim != -1 or dim != len(x.shape) - 1:
        permute = [i for i in range(len(x.shape))]
        permute[dim], permute[-1] = permute[-1], permute[dim]
        x = tf.transpose(x, permute)
        ret = tf.signal.ifft(x, operation_name)
        ret = tf.transpose(ret, permute)
        del permute
    else:
        ret = tf.signal.ifft(x, operation_name)
    ret = _ifft_norm(ret, dim, norm=norm)
    return ret


@with_unsupported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def embedding(
    weights: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    /,
    *,
    max_norm: Optional[float] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ivy.utils.assertions.check_equal(
        len(weights.shape), 2, message="weights must be 2-d", as_array=False
    )
    return tf.nn.embedding_lookup(weights, indices, max_norm=max_norm)


def interpolate(
    x: Union[tf.Tensor, tf.Variable],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "tf_bicubic",
        "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: bool = False,
    antialias: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    input_size = ivy.shape(x)[2:]
    dims = len(input_size)
    size, _ = _get_size(scale_factor, size, dims, input_size)
    if all(a == b for a, b in zip(size, input_size)):
        ret = x
    else:
        remove_dim = False
        if mode in ["linear", "tf_area", "lanczos3", "lanczos5", "nearest-exact"]:
            if dims == 1:
                size = (1,) + tuple(size)
                x = tf.expand_dims(x, axis=-2)
                dims = 2
                remove_dim = True
            mode = (
                "bilinear"
                if mode == "linear"
                else (
                    "area"
                    if mode == "tf_area"
                    else "nearest" if mode == "nearest-exact" else mode
                )
            )
        if mode == "tf_bicubic":
            mode = "bicubic"
        x = tf.transpose(x, (0, *range(2, dims + 2), 1))
        ret = tf.transpose(
            tf.cast(
                tf.image.resize(x, size=size, method=mode, antialias=antialias), x.dtype
            ),
            (0, dims + 1, *range(1, dims + 1)),
        )
        if remove_dim:
            ret = tf.squeeze(ret, axis=-2)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


interpolate.partial_mixed_handler = (
    lambda x, *args, mode="linear", recompute_scale_factor=None, align_corners=None, **kwargs: len(  # noqa: E501
        x.shape
    )
    < 4
    and mode not in ["nearest", "area", "bicubic", "nd"]
    and not align_corners
    and recompute_scale_factor
)


def _fft2_norm(
    x: Union[tf.Tensor, tf.Variable],
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
):
    n = tf.constant(s[0] * s[1], dtype=x.dtype)
    if norm == "backward":
        return x
    elif norm == "ortho":
        return x / tf.sqrt(n)
    elif norm == "forward":
        return x / n
    else:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


def trans_x_to_s(
    x: Union[tf.Tensor, tf.Variable],
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
) -> Union[tf.Tensor, tf.Variable]:
    """Change the shape of the input array x to the desired output shape s."""
    if x.dtype not in [tf.complex128, tf.complex64]:
        x = tf.cast(x, tf.float32)
    x_shape = x.shape
    if dim in [(-1, -2), (1, 0)]:
        s = (s[1], s[0])
    if s[0] >= x_shape[0] and s[1] >= x_shape[1]:
        paddings = tf.constant([[0, s[0] - x_shape[0]], [0, s[1] - x_shape[1]]])
        x_new = tf.pad(x, paddings=paddings)
    elif (s[0] <= x_shape[0] or s[1] <= x_shape[1]) and min(s) > min(x_shape):
        x_new = x[: s[0], : s[1]]
        if s[0] != x_new.shape[0]:
            size = s[0] - x_new.shape[0]
            z = tf.zeros((size, s[1]), dtype=x.dtype)
            x_new = tf.concat([x_new, z], 0)
        elif s[1] != x_new.shape[1]:
            size = s[1] - x_new.shape[1]
            z = tf.zeros((s[0], size), dtype=x.dtype)
            x_new = tf.concat([x_new, z], 1)
    elif (s[0] >= x_shape[0] and s[1] <= x_shape[1]) and min(s) <= min(x_shape):
        x_new = x[: s[0], : s[1]]
        size = s[0] - x_new.shape[0]
        z = tf.zeros((size, s[1]), dtype=x.dtype)
        x_new = tf.concat([x_new, z], 0)
    elif (s[0] < x_shape[0] and s[1] > x_shape[1]) and min(s) == min(x_shape):
        x_new = x[: s[0], : s[1]]
        size = s[1] - x_new.shape[1]
        z = tf.zeros((s[0], size), dtype=x.dtype)
        x_new = tf.concat([x_new, z], axis=1)
    else:
        x_new = x[: s[0], : s[1]]
    return x_new


def fft2_operations(x, rank):
    if x.shape.rank == 1:
        x = tf.signal.fft(x)
    elif x.shape.rank == 2:
        x = tf.switch_case(
            rank - 1, {0: lambda: tf.signal.fft(x), 1: lambda: tf.signal.fft2d(x)}
        )
    else:
        x = tf.switch_case(
            rank - 1,
            {
                0: lambda: tf.signal.fft(x),
                1: lambda: tf.signal.fft2d(x),
                2: lambda: tf.signal.fft3d(x),
            },
        )
    return x


def _fft2_helper(x, shape, axes):
    x = fft_input_validation(tf.convert_to_tensor(x))
    input_shape = x.shape
    input_rank_tensor = tf.rank(x)

    shape_, axes_ = shape_and_axes_validation(shape, axes, input_rank_tensor)

    axes = axes_initialization(shape, axes, input_shape, input_rank_tensor)

    perform_padding, perform_transpose = perform_actions_initialization(
        shape, axes, input_shape, input_rank_tensor
    )

    shape = shape_initialization(shape, axes, x)

    rank = rank_initialization(axes)

    x = get_x_after_pad_or_crop(x, shape, axes, perform_padding, input_rank_tensor)

    perm = get_perm(input_rank_tensor, axes)

    x = transpose_x(x, perm, perform_transpose)

    x = fft2_operations(x, rank)

    x = transpose_x(x, tf.argsort(perm), perform_transpose)

    x = tf.ensure_shape(x, static_output_shape(input_shape, shape_, axes_))

    return x


@with_supported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def fft2(
    x: Union[tf.Tensor, tf.Variable],
    *,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if s is None:
        s = (x.shape[dim[0]], x.shape[dim[1]])
    if len(x.shape) > 2:
        result = _fft2_helper(x, s, dim)
    else:
        x_new = trans_x_to_s(x, s, dim)
        x_complex = tf.cast(x_new, tf.complex128)
        result = tf.signal.fft2d(x_complex)

    result = _fft2_norm(result, s, dim, norm)
    if x.dtype == tf.complex64:
        result = tf.cast(result, dtype=tf.complex128)
    return result


# --- IFFTN --- #
def fft_input_validation(x):
    if not x.dtype.is_complex:
        raise TypeError(
            f"Invalid FFT input: `x` must be of a complex dtype. Received: {x.dtype}"
        )
    return x


def shape_and_axes_validation(shape, axes, input_rank_tensor):
    if shape is not None:
        shape = tf.convert_to_tensor(shape, dtype=tf.dtypes.int32)
        checks_shape = [
            tf.debugging.assert_less_equal(
                tf.size(shape),
                input_rank_tensor,
                message=(
                    "Argument `shape` cannot have length greater than the rank of `x`."
                    f" Received: {shape}"
                ),
            )
        ]
        with tf.control_dependencies(checks_shape):
            shape = tf.identity(shape)

    if axes is not None:
        axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32)
        checks_axes = [
            tf.debugging.assert_less_equal(
                tf.size(axes),
                input_rank_tensor,
                message=(
                    "Argument `axes` cannot have length greater than the rank of `x`."
                    f" Received: {axes}"
                ),
            ),
            tf.debugging.assert_less(
                axes,
                input_rank_tensor,
                message=f"Argument `axes` contains invalid indices. Received: {axes}",
            ),
            tf.debugging.assert_greater_equal(
                axes,
                -input_rank_tensor,
                message=f"Argument `axes` contains invalid indices. Received: {axes}",
            ),
        ]
        with tf.control_dependencies(checks_axes):
            axes = tf.identity(axes)

    if shape is not None and axes is not None:
        checks_shape_axes = [
            tf.debugging.assert_equal(
                tf.size(shape),
                tf.size(axes),
                message=(
                    "Arguments `shape` and `axes` must have equal length. Received:"
                    f" {shape}, {axes}"
                ),
            )
        ]
        with tf.control_dependencies(checks_shape_axes):
            shape, axes = tf.identity_n([shape, axes])

    return shape, axes


def axes_initialization(shape, axes, input_shape, input_rank_tensor):
    if axes is None:
        axes = (
            tf.range(-tf.size(input_shape), 0)
            if shape is None
            else tf.range(-tf.size(shape), 0)
        )
    axes = tf.where(tf.math.less(axes, 0), axes + input_rank_tensor, axes)
    return axes


def perform_actions_initialization(shape, axes, input_shape, input_rank_tensor):
    perform_padding = shape is not None
    perform_transpose = tf.math.logical_not(
        tf.math.reduce_all(
            tf.math.equal(
                axes, tf.range(input_rank_tensor - tf.size(axes), input_rank_tensor)
            )
        )
    )
    return perform_padding, perform_transpose


def shape_initialization(shape, axes, x):
    if shape is None:
        shape = tf.gather(tf.shape(x), axes, axis=0)
    return shape


def rank_initialization(axes):
    rank = tf.size(axes)
    with tf.control_dependencies(
        [
            tf.debugging.assert_less_equal(
                rank, 3, message="N-D FFT supported only up to 3-D."
            )
        ]
    ):
        rank = tf.identity(rank)

    return rank


def norm_initialization(norm, shape, x):
    if norm == "backward":
        norm_factor = tf.constant(1, x.dtype)
    elif norm in ["forward", "ortho"]:
        norm_factor = tf.cast(tf.math.reduce_prod(shape), x.dtype)
        if norm == "ortho":
            norm_factor = tf.math.sqrt(norm_factor)
    return norm_factor


def get_x_after_pad_or_crop(x, shape, axes, perform_padding, input_rank_tensor):
    if perform_padding:
        pad_shape = -tf.ones([input_rank_tensor], dtype=tf.int32)
        pad_shape = tf.tensor_scatter_nd_update(
            pad_shape, tf.expand_dims(axes, -1), shape
        )
        x = _right_pad_or_crop(x, pad_shape)
    return x


def get_perm(input_rank_tensor, axes):
    all_dims = tf.range(input_rank_tensor, dtype=tf.dtypes.int32)
    perm = tf.concat(
        [
            tf.boolean_mask(
                all_dims,
                tf.foldl(
                    lambda acc, elem: tf.math.logical_and(
                        acc, tf.math.not_equal(all_dims, elem)
                    ),
                    axes,
                    initializer=tf.fill(all_dims.shape, True),
                ),
            ),
            axes,
        ],
        0,
    )
    return perm


def ifft_operations(x, rank, norm_factor):
    if x.shape.rank == 1:
        x = tf.signal.ifft(x)
    elif x.shape.rank == 2:
        x = tf.switch_case(
            rank - 1, {0: lambda: tf.signal.ifft(x), 1: lambda: tf.signal.ifft2d(x)}
        )
    else:
        x = tf.switch_case(
            rank - 1,
            {
                0: lambda: tf.signal.ifft(x),
                1: lambda: tf.signal.ifft2d(x),
                2: lambda: tf.signal.ifft3d(x),
            },
        )
    x = x * norm_factor
    return x


def transpose_x(x, perm, perform_transpose):
    x = tf.cond(perform_transpose, lambda: tf.transpose(x, perm=perm), lambda: x)
    return x


def static_output_shape(input_shape, shape, axes):
    output_shape = input_shape.as_list()
    if shape is not None:
        if axes is None:
            axes = list(range(-len(shape), 0))
        if isinstance(shape, tf.Tensor):
            if isinstance(axes, tf.Tensor):
                output_shape = [None] * len(output_shape)
            else:
                for ax in axes:
                    output_shape[ax] = None
        else:
            for idx, ax in enumerate(axes):
                output_shape[ax] = shape[idx]
    return tf.TensorShape(output_shape)


def _right_pad_or_crop(tensor, shape):
    input_shape = tf.shape(tensor)
    shape = tf.convert_to_tensor(shape, dtype=tf.dtypes.int32)
    with tf.control_dependencies(
        [tf.debugging.assert_less_equal(tf.size(shape), tf.size(input_shape))]
    ):
        shape = tf.identity(shape)
    shape = tf.concat([input_shape[: tf.size(input_shape) - tf.size(shape)], shape], 0)

    pad_sizes = tf.math.maximum(shape - input_shape, 0)
    pad_sizes = tf.expand_dims(pad_sizes, -1)
    pad_sizes = tf.concat(
        [tf.zeros(pad_sizes.shape, dtype=tf.dtypes.int32), pad_sizes], -1
    )
    tensor = tf.pad(tensor, pad_sizes, constant_values=0)

    crop_tensor = tf.zeros(shape.shape, dtype=tf.dtypes.int32)
    tensor = tf.slice(tensor, crop_tensor, shape)
    return tensor


def _ifftn_helper(x, shape, axes, norm):
    x = fft_input_validation(tf.convert_to_tensor(x))
    input_shape = x.shape
    input_rank_tensor = tf.rank(x)

    shape_, axes_ = shape_and_axes_validation(shape, axes, input_rank_tensor)

    axes = axes_initialization(shape, axes, input_shape, input_rank_tensor)

    perform_padding, perform_transpose = perform_actions_initialization(
        shape, axes, input_shape, input_rank_tensor
    )

    shape = shape_initialization(shape, axes, x)

    rank = rank_initialization(axes)

    norm_factor = norm_initialization(norm, shape, x)

    x = get_x_after_pad_or_crop(x, shape, axes, perform_padding, input_rank_tensor)

    perm = get_perm(input_rank_tensor, axes)

    x = transpose_x(x, perm, perform_transpose)

    x = ifft_operations(x, rank, norm_factor)

    x = transpose_x(x, tf.argsort(perm), perform_transpose)

    x = tf.ensure_shape(x, static_output_shape(input_shape, shape_, axes_))

    return x


def ifftn(
    x: Union[tf.Tensor, tf.Variable],
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: Optional[str] = "backward",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    result = _ifftn_helper(x, s, axes, norm)

    if out is not None:
        out = result
        return out
    else:
        return result


"""
RFFTN Function
"""


def rfft_input_validation(x):
    if not x.dtype.is_floating:
        raise TypeError(
            f"Invalid FFT input: `x` must be of a real dtype. Received: {x.dtype}"
        )
    return x


def rfft_operations(x, rank, norm_factor):
    if x.shape.rank == 1:
        x = tf.signal.rfft(x)
    elif x.shape.rank == 2:
        x = tf.switch_case(
            rank - 1, {0: lambda: tf.signal.rfft(x), 1: lambda: tf.signal.rfft2d(x)}
        )
    else:
        x = tf.switch_case(
            rank - 1,
            {
                0: lambda: tf.signal.rfft(x),
                1: lambda: tf.signal.rfft2d(x),
                2: lambda: tf.signal.rfft3d(x),
            },
        )
    norm_factor = tf.cast(norm_factor, tf.complex128)
    x = tf.cast(x, tf.complex128)
    x = x / norm_factor
    return x


def _rfftn_helper(x, shape, axes, norm):
    x = rfft_input_validation(tf.convert_to_tensor(x))
    input_shape = x.shape
    input_rank_tensor = tf.rank(x)

    shape_, axes_ = shape_and_axes_validation(shape, axes, input_rank_tensor)

    axes = axes_initialization(shape, axes, input_shape, input_rank_tensor)

    perform_padding, perform_transpose = perform_actions_initialization(
        shape, axes, input_shape, input_rank_tensor
    )

    shape = shape_initialization(shape, axes, x)

    rank = rank_initialization(axes)

    norm_factor = norm_initialization(norm, shape, x)

    x = get_x_after_pad_or_crop(x, shape, axes, perform_padding, input_rank_tensor)

    perm = get_perm(input_rank_tensor, axes)

    x = transpose_x(x, perm, perform_transpose)

    x = rfft_operations(x, rank, norm_factor)

    x = transpose_x(x, tf.argsort(perm), perform_transpose)

    x = tf.ensure_shape(x, static_output_shape(input_shape, shape_, axes_))

    return x


def rfft(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # type cast
    if x.dtype in [tf.complex64, tf.complex128]:
        x = tf.math.real(x)
    if x.dtype not in [tf.float32, tf.float64]:
        x = tf.cast(x, tf.float32)

    # axis check
    if not isinstance(axis, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(axis)}"
        )

    # axis normalization
    naxis = axis
    if axis < 0:
        naxis = x.ndim + axis
    if naxis < 0 or naxis >= x.ndim:
        raise ivy.utils.exceptions.IvyError(
            f"Axis {axis} is out of bounds for array of dimension {x.ndim}"
        )
    axis = naxis

    # n checks
    if n is None:
        n = x.shape[axis]
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n < 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid number of FFT data points ({n}) specified."
        )

    # norm check & value
    if norm == "backward":
        inv_norm = tf.constant(1, dtype=x.dtype)
    elif norm in ["forward", "ortho"]:
        inv_norm = tf.cast(tf.math.reduce_prod(n), dtype=x.dtype)
        if norm == "ortho":
            inv_norm = tf.math.sqrt(inv_norm)
    else:
        raise ivy.utils.exceptions.IvyError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        )
    fct = 1 / inv_norm

    if x.shape[axis] != n:
        s = list(x.shape)
        if s[axis] > n:
            index = [slice(None)] * len(s)
            index[axis] = slice(0, n)
            x = x[tuple(index)]
        else:
            s[axis] = n - s[axis]
            z = tf.zeros(s, x.dtype)
            x = tf.concat([x, z], axis=axis)

    if axis == x.ndim - 1:
        ret = tf.signal.rfft(x, fft_length=None, name=None)
    else:
        x = tf.experimental.numpy.swapaxes(x, axis, -1)
        ret = tf.signal.rfft(x, fft_length=None, name=None)
        ret = tf.experimental.numpy.swapaxes(ret, axis, -1)

    ret *= tf.cast(fct, dtype=ret.dtype)

    if x.dtype != tf.float64:
        ret = tf.cast(ret, dtype=tf.complex64)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@with_supported_device_and_dtypes(
    {
        "2.5.0 and above": {
            "cpu": (
                "float32",
                "float64",
                "complex128",
            )
        }
    },
    backend_version,
)
def rfftn(
    x: Union[tf.Tensor, tf.Variable],
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: str = "backward",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    result = _rfftn_helper(x, s, axes, norm)

    if out is not None:
        out = tf.cast(result, tf.complex128)
        # out = result
        return out
    else:
        # return result
        return tf.cast(result, tf.complex128)


# stft
@with_supported_dtypes({"2.15.0 and below": ("complex",)}, backend_version)
def stft(
    signals: Union[tf.Tensor, tf.Variable],
    frame_length: int,
    frame_step: int,
    /,
    *,
    fft_length: Optional[int] = None,
    window_fn: Optional[Callable] = None,
    pad_end: Optional[bool] = False,
    name: Optional[str] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not isinstance(frame_length, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_length)}"
        )

    if frame_length < 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {frame_length}, expecting frame_length larger than or"
            " equal to 1"
        )

    if not isinstance(frame_step, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(frame_step)}"
        )

    if frame_step < 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {frame_length}, expecting frame_length larger than or"
            " equal to 1"
        )

    if fft_length is not None:
        if not isinstance(fft_length, int):
            raise ivy.utils.exceptions.IvyError(
                f"Expecting <class 'int'> instead of {type(fft_length)}"
            )

        if fft_length < 1:
            raise ivy.utils.exceptions.IvyError(
                f"Invalid data points {frame_length}, expecting frame_length larger"
                " than or equal to 1"
            )

    result = tf.signal.stft(
        signals,
        frame_length,
        frame_step,
        fft_length=fft_length,
        window_fn=window_fn,
        pad_end=pad_end,
        name=name,
    )

    if out is not None:
        return out

    else:
        return result


def _to_4d(x):
    t = x  # Start with the original tensor
    while len(t.shape) < 4:  # Continue expanding dimensions until 4D
        t = tf.expand_dims(t, axis=0)
    return t


def sliding_window(
    input: Union[tf.Tensor, tf.Variable],
    kernel_size: Union[int, Tuple[int, int]],
    /,
    *,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = "VALID",
) -> Union[tf.Tensor, tf.Variable]:
    if len(input.shape) != 4:
        input = _to_4d(input)

    input = tf.transpose(input, (0, 2, 3, 1))

    kernel_size = (
        [1]
        + ([kernel_size] * 2 if isinstance(kernel_size, int) else list(kernel_size))
        + [1]
    )
    if len(kernel_size) < 4:
        kernel_size.append(1)

    stride = [1] + ([stride] * 2 if isinstance(stride, int) else list(stride)) + [1]
    if len(stride) < 4:
        stride.append(1)

    dilation = (
        [1] + ([dilation] * 2 if isinstance(dilation, int) else list(dilation)) + [1]
    )
    if len(dilation) < 4:
        dilation.append(1)

    padding = [padding] * 2 if isinstance(padding, int) else padding

    if isinstance(padding, str) and padding.upper() in ["VALID", "SAME"]:
        padding = padding

    elif padding[0] == padding[1] == 0:
        padding = "VALID"
    elif padding[0] == padding[1] != 0:
        padding = "SAME"
    else:
        raise ivy.utils.exceptions.IvyError(
            f"Cannot convert padding sequence {padding} to TensorFlow padding mode"
        )

    return tf.image.extract_patches(
        images=input, sizes=kernel_size, strides=stride, rates=dilation, padding=padding
    )


def rnn(
    step_function,
    inputs,
    initial_states,
    /,
    *,
    go_backwards: bool = False,
    mask: Optional[Union[tf.Tensor, tf.Variable]] = None,
    constants: Optional[Union[tf.Tensor, tf.Variable]] = None,
    unroll: bool = False,
    input_length: Optional[int] = None,
    time_major: bool = False,
    zero_output_for_mask: bool = False,
    return_all_outputs: bool = True,
):
    step_function = inputs_to_ivy_arrays(output_to_native_arrays(step_function))
    return tf.keras.backend.rnn(
        step_function,
        inputs,
        initial_states,
        go_backwards=go_backwards,
        mask=mask,
        constants=constants,
        unroll=unroll,
        input_length=input_length,
        time_major=time_major,
        zero_output_for_mask=zero_output_for_mask,
        return_all_outputs=return_all_outputs,
    )
