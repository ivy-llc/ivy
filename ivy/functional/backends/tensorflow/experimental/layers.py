# global
import math
from typing import Union, Optional, Tuple, Literal, Sequence
import tensorflow as tf

# local
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from .. import backend_version
import ivy
from ivy.functional.ivy.layers import _handle_padding, _get_num_padded_values
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode, _get_size


def _from_int_to_tuple(arg, dim):
    if isinstance(arg, int):
        return (arg,) * dim
    if isinstance(arg, (tuple, list)) and len(arg) == 1:
        return (arg[0],) * dim
    return arg


def _determine_depth_max_pooling(x, kernel, strides, dims):
    # determine depth pooling
    depth_pooling = False
    if len(kernel) == dims + 2:
        spatial_kernel = kernel[1:-1]
        if kernel[-1] != 1:
            depth_pooling = True
            if any(tf.constant(spatial_kernel) != 1):
                raise NotImplementedError(
                    "MaxPooling supports exactly one of pooling across"
                    " depth or pooling across width/height."
                )
            if len(strides) != dims + 2 or strides[-1] != kernel[-1]:
                raise NotImplementedError(
                    "Depthwise max pooling requires the depth window to equal the depth"
                    " stride"
                )
            if x.shape[-1] % kernel[-1] != 0:
                raise NotImplementedError(
                    "Depthwise max pooling requires the depth window to evenly divide"
                    " the input depth"
                )
            x = tf.transpose(x, (0, dims + 1, *range(1, dims + 1)))
            kernel = [kernel[-1], *[1] * (dims - 1)]
            strides = [strides[-1], *[1] * (dims - 1)]
        else:
            kernel = spatial_kernel
            strides = strides[1:-1] if len(strides) == dims + 2 else strides
    return x, kernel, strides, depth_pooling


def max_pool1d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    res = tf.nn.max_pool1d(x, kernel, strides, padding)

    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


def max_pool2d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, Tuple[int], Tuple[int, int]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))

    dilation = _from_int_to_tuple(dilation, 2)
    strides = _from_int_to_tuple(strides, 2)
    kernel = _from_int_to_tuple(kernel, 2)

    # determine depth pooling
    x, kernel, strides, depth_pooling = _determine_depth_max_pooling(
        x, kernel, strides, 2
    )

    if not depth_pooling:
        if isinstance(padding, int):
            padding = [(padding,) * 2] * 2
        elif isinstance(padding, tuple) and len(padding) == 1:
            padding = [(padding[0],) * 2] * 2
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding = [(padding[0],) * 2, (padding[1],) * 2]

        if isinstance(padding, (tuple, list)):
            ivy.utils.assertions.check_kernel_padding_size(kernel, padding)
        new_kernel = [kernel[i] + (kernel[i] - 1) * (dilation[i] - 1) for i in range(2)]
        if isinstance(padding, str):
            pad_h = _handle_padding(x.shape[1], strides[0], new_kernel[0], padding)
            pad_w = _handle_padding(x.shape[2], strides[1], new_kernel[1], padding)
            padding = [
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ]

        x_shape = x.shape[1:-1]

        if ceil_mode:
            for i in range(2):
                padding[i] = _padding_ceil_mode(
                    x_shape[i], new_kernel[i], padding[i], strides[i]
                )
        padding = [(0, 0)] + list(padding) + [(0, 0)]
        x = tf.pad(x, padding, constant_values=-math.inf)
    res = tf.nn.pool(x, kernel, "MAX", strides, "VALID", dilations=dilation)

    if depth_pooling:
        res = tf.transpose(res, (0, 2, 3, 1))
    # converting minimum value to -inf because tensorflow clips -inf to minimum value
    res = tf.where(res <= ivy.finfo(res.dtype).min, -math.inf, res)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes(
    {"2.13.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def max_pool3d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    res = tf.nn.max_pool3d(x, kernel, strides, padding)
    if data_format == "NCDHW":
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


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16", "float64")}, backend_version)
def avg_pool1d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
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
                tf.constant([c], dtype=res.dtype),
                tf.constant([res.shape[1]], dtype=tf.int32),
            )
        res = (kernel[0] * res) / (kernel[0] - num_padded_values[:, None])

    if data_format in ("NCW", "NCL"):
        res = tf.transpose(res, (0, 2, 1))
    return res


@with_unsupported_dtypes(
    {"2.13.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def avg_pool2d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(kernel, int):
        kernel = [kernel] * 2
    elif len(kernel) == 1:
        kernel = [kernel[0]] * 2

    if isinstance(strides, int):
        strides = [strides] * 2
    elif len(strides) == 1:
        strides = [strides[0]] * 2

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
    {"2.13.0 and below": ("bfloat16", "float64", "float16")}, backend_version
)
def avg_pool3d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if isinstance(kernel, int):
        kernel = [kernel] * 3
    elif len(kernel) == 1:
        kernel = [kernel[0]] * 3

    if isinstance(strides, int):
        strides = [strides] * 3
    elif len(strides) == 1:
        strides = [strides[0]] * 3

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
            strides,
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
    {"2.13.0 and below": ("bfloat16", "float64", "float16")}, backend_version
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


@with_supported_dtypes({"2.13.0 and below": ("float32", "float64")}, backend_version)
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


@with_supported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def fft(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Union[int, Tuple[int]] = None,
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
    if norm != "backward" and norm != "ortho" and norm != "forward":
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
    x = ivy.astype(x, dtype) if dtype else x
    res = tf.nn.dropout(x, prob, noise_shape=noise_shape, seed=seed) if training else x
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
    n: Union[int, Tuple[int]] = None,
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
    if norm != "backward" and norm != "ortho" and norm != "forward":
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


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def embedding(
    weights: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    /,
    *,
    max_norm: Optional[float] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
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
        "nearest",
        "area",
        "nearest-exact",
        "tf_area",
        "bicubic",
        "bicubic_tensorflow",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: Optional[bool] = None,
    antialias: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    dims = len(x.shape) - 2
    size = _get_size(scale_factor, size, dims, x.shape)
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
    if mode == "bicubic_tensorflow":
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
    return ret

    
interpolate.partial_mixed_handler = lambda x, *args, mode="linear", scale_factor=None, recompute_scale_factor=None, align_corners=None, **kwargs: (  # noqa: E501
    (not align_corners and (len(x.shape) - 2) < 2)
    and mode not in ["nearest", "area", "bicubic", "nd"]
)


def _fft2_norm(
    x: Union[tf.Tensor, tf.Variable],
    s: Sequence[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
):
    n = tf.constant(s[0] * s[1], dtype=tf.complex128)
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
    s: Sequence[int] = None,
    dim: Sequence[int] = (-2, -1),
) -> Union[tf.Tensor, tf.Variable]:
    """change the shape of the input array x to the desired output shape s."""
    if x.dtype != tf.complex128 or x.dtype != tf.complex64:
        x = tf.cast(x, tf.float32)
    x_shape = x.shape
    if dim == (-1, -2) or dim == (1, 0):
        s = (s[1], s[0])
    if s[0] >= x_shape[0] and s[1] >= x_shape[1]:
        paddings = tf.constant([[0, s[0] - x_shape[0]], [0, s[1] - x_shape[1]]])
        x_new = tf.pad(x, paddings=paddings)
    elif (s[0] <= x_shape[0] or s[1] <= x_shape[1]) and min(s) > min(x_shape):
        x_new = x[: s[0], : s[1]]
        if s[0] != x_new.shape[0]:
            size = s[0] - x_new.shape[0]
            z = tf.zeros((size, s[1]))
            x_new = tf.concat([x_new, z], 0)
        elif s[1] != x_new.shape[1]:
            size = s[1] - x_new.shape[1]
            z = tf.zeros((s[0], size))
            x_new = tf.concat([x_new, z], 1)
    elif (s[0] >= x_shape[0] and s[1] <= x_shape[1]) and min(s) <= min(x_shape):
        x_new = x[: s[0], : s[1]]
        size = s[0] - x_new.shape[0]
        z = tf.zeros((size, s[1]))
        x_new = tf.concat([x_new, z], 0)
    elif (s[0] < x_shape[0] and s[1] > x_shape[1]) and min(s) == min(x_shape):
        x_new = x[: s[0], : s[1]]
        size = s[1] - x_new.shape[1]
        z = tf.zeros((s[0], size))
        x_new = tf.concat([x_new, z], axis=1)
    else:
        x_new = x[: s[0], : s[1]]
    return x_new


def fft2(
    x: Union[tf.Tensor, tf.Variable],
    *,
    s: Sequence[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if s is None:
        s = (x.shape[dim[0]], x.shape[dim[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    operation_name = f"{s} points FFT at dim {dim} with {norm} normalization"
    if len(x.shape) == 2:
        x_new = trans_x_to_s(x, s, dim)
        x_complex = tf.cast(x_new, tf.complex128)
        tf_fft2 = tf.signal.fft2d(x_complex, name=operation_name)
    elif len(x.shape) > 2:
        x_s = [trans_x_to_s(x[:, :, i], s, dim) for i in range(x.shape[2])]
        x_new = tf.convert_to_tensor(x_s, dtype=x.dtype)
        x_complex = tf.cast(x_new, tf.complex128)
        tf_fft2 = tf.transpose(tf.signal.fft2d(x_complex, name=operation_name))

    # Apply the same normalization as 'backward' in NumPy
    tf_fft2 = _fft2_norm(tf_fft2, s, dim, norm)
    return tf_fft2


# --- IFFTN --- #
def fft_input_validation(x):
    if not x.dtype.is_complex:
        raise TypeError(
            "Invalid FFT input: `x` must be of a complex dtype. Received: {}".format(
                x.dtype
            )
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
                    "Argument `shape` cannot have length greater than the rank of `x`. "
                    "Received: {}"
                ).format(shape),
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
                    "Argument `axes` cannot have length greater than the rank of `x`. "
                    "Received: {}"
                ).format(axes),
            ),
            tf.debugging.assert_less(
                axes,
                input_rank_tensor,
                message=(
                    "Argument `axes` contains invalid indices. Received: {}"
                ).format(axes),
            ),
            tf.debugging.assert_greater_equal(
                axes,
                -input_rank_tensor,
                message=(
                    "Argument `axes` contains invalid indices. Received: {}"
                ).format(axes),
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
                    "Arguments `shape` and `axes` must have equal length. "
                    "Received: {}, {}"
                ).format(shape, axes),
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
    elif norm == "forward" or norm == "ortho":
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


def stft(
    signal: Union[tf.Tensor, int, Tuple[int]],
    n_fft: Union[int, Tuple[int]],
    frame_step: int,
    /,
    *,
    axis: Optional[int] = None,
    onesided:Optional[bool] = True,
    fs: Optional[float] = 1.0,
    window: Optional[Union[tf.Tensor, list, str, Tuple[int]]] = None,
    win_length: Optional[int] = None,
    noverlap: Optional[int] = None,
    center: Optional[bool] = True,
    pad_mode: Optional[str] = "reflect",
    normalized: Optional[bool] = False,
    detrend: Optional[Union[str, callable, bool]] = False,
    return_complex: Optional[bool] = True,
    boundary: Optional[str] = None,
    out: Optional[tf.Tensor] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if not isinstance(n_fft, int):
        raise TypeError("n_fft must be an int.")

    return tf.signal.stft(
        signal,
        n_fft,
        frame_step,
        window,
        win_length,
        pad_mode,
    )
"""
RFFTN Function implementation
"""


def rfft_input_validation(x):
    if not x.dtype.is_floating:
        raise TypeError(
            "Invalid FFT input: `x` must be of a real dtype. Received: {}".format(
                x.dtype
            )
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
    # norm_factor = tf.cast(norm_factor, tf.complex64)
    norm_factor = tf.cast(norm_factor, tf.complex128)
    x = x * norm_factor
    return x


def _rfftn_helper(x, shape, axes, norm):
    x = rfft_input_validation(tf.convert_to_tensor(x))
    # x = rfft_input_validation(x)
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
    norm: Optional[str] = [("forward", "ortho", "backward")],
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


# def _rfftn_norm(
#     x: Union[tf.Tensor, tf.Variable],
#     s: Sequence[int] = None,
#     axes: Sequence[int] = None,
#     norm: str = "backward",
# ):
#     n = tf.constant(s[-1] // 2 + 1, dtype=tf.complex128)
#     if norm == "backward":
#         return x
#     elif norm == "ortho":
#         return x / tf.sqrt(n)
#     elif norm == "forward":
#         return x / n
#     else:
#         raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


# @with_unsupported_dtypes(
#     {"0.4.14 and below": ("float32", "complex")}, backend_version
# )
# def rfftn(
#     x: Union[tf.Tensor, tf.Variable],
#     s: Sequence[int] = None,
#     axes: Sequence[int] = None,
#     *,
#     norm: str = "backward",
#     out: Optional[Union[tf.Tensor, tf.Variable]] = None,
# ) -> Union[tf.Tensor, tf.Variable]:
#     if axes is None:
#         axes = list(range(len(s)))
#     elif s is None:
#         s = [x.shape[axis] for axis in axes]
#     elif len(s) != len(axes):
#         raise ValueError("s and axes must have the same length.")

#     if not all(isinstance(j, int) for j in s):
#         raise ivy.utils.exceptions.IvyError(
#             f"Expecting {s} to be a sequence of integers <class integer>"
#         )
#     if all(j <= 1 for j in s):
#         raise ivy.utils.exceptions.IvyError(
#             f"Invalid data points {s}, expecting s points larger than 1"
#         )
#     if norm != "backward" and norm != "ortho" and norm != "forward":
#         raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")


#     if len(s) != 2:
#         raise ValueError("fft_length must have shape [2]")
#     fft_length = tf.convert_to_tensor(s, dtype=tf.int32)

#     # Convert s to a tensor with shape [2]
#     # fft_length = tf.convert_to_tensor(s, dtype=tf.int32)
#     # if tf.shape(fft_length).shape != [1] or tf.shape(fft_length).as_list()[0] != 2:
#     #     raise ValueError("fft_length must have shape [2]")
#     operation_name = f"RFFTn with s={s}, axes={axes}, norm={norm}"
#     # tf_rfftn = tf.signal.rfft2d(x, fft_length=fft_length, name=operation_name)

# # Reshape the input tensor to 2D
#     x_2d = tf.reshape(x, [-1, s[0]])

#     # Perform rfft2d operation on the reshaped tensor
#     tf_rfftn_2d = tf.signal.rfft2d(x_2d, fft_length=s[1], name=operation_name)

#     # Reshape the result back to the original shape
#     output_shape = tf.concat([tf.shape(x)[:-1], [s[-1] // 2 + 1]], axis=0)
#     tf_rfftn = tf.reshape(tf_rfftn_2d, output_shape)


#     # Apply the same normalization as 'backward' in NumPy
#     tf_rfftn = _rfftn_norm(tf_rfftn, s, axes, norm, out).astype("complex128")
#     return tf_rfftn
