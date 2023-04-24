# global
import math
from typing import Union, Optional, Tuple, Literal, Sequence
import tensorflow as tf

# local
from ivy.func_wrapper import with_unsupported_dtypes, handle_mixed_function
from .. import backend_version
import ivy
from ivy.functional.ivy.layers import _handle_padding, _get_num_padded_values
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode, _get_size


def _from_int_to_tuple(arg, dim):
    if isinstance(arg, int):
        return (arg,) * dim
    if isinstance(arg, tuple) and len(arg) == 1:
        return (arg[0],) * dim
    return arg


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
        padding = [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]

    x_shape = x.shape[1:-1]

    if ceil_mode:
        for i in range(2):
            padding[i] = _padding_ceil_mode(
                x_shape[i], new_kernel[i], padding[i], strides[i]
            )

    padding = [(0, 0)] + list(padding) + [(0, 0)]
    x = tf.pad(x, padding, constant_values=-math.inf)
    res = tf.nn.pool(x, kernel, "MAX", strides, "VALID", dilations=dilation)

    # converting minimum value to -inf because tensorflow clips -inf to minimum value
    res = tf.where(res <= ivy.finfo(res.dtype).min, -math.inf, res)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


@with_unsupported_dtypes(
    {"2.9.1 and below": ("bfloat16", "float64", "float16")}, backend_version
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


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16", "float64")}, backend_version)
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

    if data_format == "NCW":
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

    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
    return res


@with_unsupported_dtypes(
    {"2.9.1 and below": ("bfloat16", "float64", "float16")}, backend_version
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
            x, tf.ones(kernel + [x.shape[-1], 1]), [1] + strides + [1], padding
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
    {"2.9.1 and below": ("bfloat16", "float64", "float16")}, backend_version
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
    if x.dtype not in (tf.float32, tf.float64):
        x = tf.cast(x, tf.float32)
    if axis != -1:
        new_dims = list(range(len(x.shape)))
        new_dims[axis], new_dims[-1] = new_dims[-1], axis
        x = tf.transpose(x, new_dims)
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
        dct_out = tf.transpose(dct_out, new_dims)
    else:
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
    return dct_out


def _fft_norm(
    x: Union[tf.Tensor, tf.Variable],
    dim: int,
    /,
    *,
    norm: str = "backward",
):
    n = tf.constant(x.shape[dim])
    if norm == "backward":
        return x
    elif norm == "ortho":
        return x / tf.sqrt(n)
    elif norm == "forward":
        return x / n
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
            x = tf.concat([x, z], axis=dim)
        del s
    operation_name = f"{n} points FFT at dim {dim} with {norm} normalization"
    if dim != -1 or dim != len(x.shape) - 1:
        permute = [i for i in range(len(x.shape))]
        permute[dim], permute[-1] = permute[-1], permute[dim]
        x = tf.transpose(x, permute)
        ret = tf.signal.fft(x, operation_name)
        x = tf.transpose(x, permute)
        del permute
    else:
        ret = tf.signal.fft(x, operation_name)
    ret = _fft_norm(ret, dim, norm=norm)
    return ret


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
        if data_format == "NCW":
            perm = (0, 2, 1) if len(x.shape) == 3 else (1, 0)
            x = tf.transpose(x, perm)
        noise_shape = list(x.shape)
        noise_shape[-2] = 1
        res = tf.nn.dropout(x, prob, noise_shape=noise_shape)
        if data_format == "NCW":
            res = tf.transpose(res, perm)
        return res
    else:
        return x


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
        noise_shape = list(x.shape)
        sl = slice(1, -1) if is_batched else slice(-1)
        noise_shape[sl] = [1] * 3
        res = tf.nn.dropout(x, prob, noise_shape=noise_shape)
        if data_format == "NCDHW":
            perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
            res = tf.transpose(res, perm)
        return res
    else:
        return x


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


def embedding(
    weights: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    /,
    *,
    max_norm: Optional[float] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.nn.embedding_lookup(weights, indices, max_norm=max_norm)


@handle_mixed_function(
    lambda x, *args, mode="linear", scale_factor=None, recompute_scale_factor=None, align_corners=None, **kwargs: (  # NOQA
        not align_corners and (len(x.shape) - 2) < 2
    )
    and mode not in ["nearest", "area", "bicubic"]
    and recompute_scale_factor
)
def interpolate(
    x: Union[tf.Tensor, tf.Variable],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Union[
        Literal[
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
        ]
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
            else "area"
            if mode == "tf_area"
            else "nearest"
            if mode == "nearest-exact"
            else mode
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
