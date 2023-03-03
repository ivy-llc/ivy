# global
import math
from typing import Union, Optional, Tuple, Literal, Sequence
import tensorflow as tf

# local
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version
import ivy
from ivy.functional.ivy.layers import _handle_padding
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode


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


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16", "float64")}, backend_version)
def avg_pool1d(
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
    res = tf.nn.avg_pool1d(x, kernel, strides, padding)

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
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.avg_pool2d(x, kernel, strides, padding)
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
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    res = tf.nn.avg_pool3d(x, kernel, strides, padding)
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res


def dct(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
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
    norm: Optional[str] = "backward",
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
    norm: Optional[str] = "backward",
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


def interpolate(
    x: Union[tf.Tensor, tf.Variable],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Union[
        Literal["linear", "bilinear", "trilinear", "nearest", "area"]
    ] = "linear",
    align_corners: Optional[bool] = None,
    antialias: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    dims = len(x.shape) - 2
    if align_corners or dims > 2 or mode in ["nearest", "area"]:
        return ivy.functional.experimental.interpolate(
            x, size, mode=mode, align_corners=align_corners, antialias=antialias
        )
    size = (size,) * dims if isinstance(size, int) else size
    remove_dim = False
    if mode in ["linear", "area"]:
        if dims == 1:
            size = (1,) + tuple(size)
            x = tf.expand_dims(x, axis=-2)
            dims = 2
            remove_dim = True
        mode = "bilinear" if mode == "linear" else mode
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
