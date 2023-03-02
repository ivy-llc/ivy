# global
from typing import Optional, Union, Tuple, Literal, Sequence
import jax
import jax.lax as jlax
import jax.numpy as jnp
import math

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.random import RNG
from ivy.functional.ivy.layers import _handle_padding
from ivy.functional.ivy.experimental.layers import _padding_ceil_mode


def _from_int_to_tuple(arg, dim):
    if isinstance(arg, int):
        return (arg,) * dim
    if isinstance(arg, tuple) and len(arg) == 1:
        return (arg[0],) * dim
    return arg


def general_pool(
    inputs, init, reduce_fn, window_shape, strides, padding, dim, dilation=1, ceil_mode=False,
):
    window_shape = _from_int_to_tuple(window_shape, dim)
    strides = _from_int_to_tuple(strides, dim)
    dilation = _from_int_to_tuple(dilation, dim)
    if isinstance(padding, int):
        padding = [(padding,) * 2] * dim
    elif isinstance(padding, tuple) and len(padding) == 1:
        padding = [(padding[0],) * 2] * dim
    elif isinstance(padding, tuple) and len(padding) == 2:
        padding = [(padding[0],) * 2, (padding[1],) * 2]

    if isinstance(padding, (tuple, list)):
        ivy.utils.assertions.check_kernel_padding_size(window_shape, padding)

    assert len(window_shape) == len(
        strides
    ), f"len({window_shape}) must equal len({strides})"

    window_shape = tuple(window_shape)
    strides = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)
    dilation = (1,) + tuple(dilation) + (1,)

    is_single_input = False
    if inputs.ndim == len(dims) - 1:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"

    # shape of window after dilation
    new_window_shape = tuple(
        [
            window_shape[i - 1] + (dilation[i] - 1) * (window_shape[i - 1] - 1)
            for i in range(1, len(dims) - 1)
        ]
    )
    # manual padding
    if isinstance(padding, str):
        pad_int = [
            _handle_padding(
                inputs.shape[i + 1], strides[i + 1], new_window_shape[i], padding
            )
            for i in range(len(dims) - 2)
        ]
        pad_list = [
            (pad_int[i] // 2, pad_int[i] - pad_int[i] // 2) for i in range(len(pad_int))
        ]
        pad_list = [(0, 0)] + pad_list + [(0, 0)]
    else:
        pad_list = [(0, 0)] + list(padding) + [(0, 0)]

    if ceil_mode:
        for i in range(len(dims) - 2):
            pad_list[i + 1] = _padding_ceil_mode(
                inputs.shape[i + 1],
                new_window_shape[i],
                pad_list[i + 1],
                strides[i + 1],
            )

    y = jlax.reduce_window(
        inputs, init, reduce_fn, dims, strides, pad_list, window_dilation=dilation
    )
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def max_pool1d(
    x: JaxArray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if data_format == "NCW":
        x = jnp.transpose(x, (0, 2, 1))

    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    res = general_pool(x, -jnp.inf, jlax.max, kernel, strides, padding, 1)

    if data_format == "NCW":
        res = jnp.transpose(x, (0, 2, 1))
    return res


def max_pool2d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, Tuple[int], Tuple[int, int]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if data_format == "NCHW":
        x = jnp.transpose(x, (0, 2, 3, 1))

    res = general_pool(
        x, -jnp.inf, jlax.max, kernel, strides, padding, 2, dilation, ceil_mode
    )

    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))

    return res


def max_pool3d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if data_format == "NCDHW":
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
    if isinstance(kernel, int):
        kernel = (kernel,) * 3
    res = general_pool(x, -jnp.inf, jlax.max, kernel, strides, padding, 3)

    if data_format == "NCDHW":
        res = jnp.transpose(x, (0, 2, 3, 4, 1))

    return res


def avg_pool1d(
    x: JaxArray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if data_format == "NCW":
        x = jnp.transpose(x, (0, 2, 1))

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    res = general_pool(x, 0.0, jlax.add, kernel, strides, padding, 1)
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / general_pool(
        jnp.ones(div_shape, dtype=res.dtype), 0.0, jlax.add, kernel, strides, padding, 1
    )
    if data_format == "NCW":
        res = jnp.transpose(res, (0, 2, 1))
    return res


def avg_pool2d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if isinstance(kernel, int):
        kernel = (kernel,) * 2
    elif len(kernel) == 1:
        kernel = (kernel[0],) * 2

    if isinstance(strides, int):
        strides = (strides,) * 2
    elif len(strides) == 1:
        strides = (strides[0],) * 2

    if data_format == "NCHW":
        x = jnp.transpose(x, (0, 2, 3, 1))

    res = general_pool(x, 0.0, jlax.add, kernel, strides, padding, 2)
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / general_pool(
        jnp.ones(div_shape, dtype=res.dtype), 0.0, jlax.add, kernel, strides, padding, 2
    )
    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))
    return res


def avg_pool3d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if isinstance(kernel, int):
        kernel = (kernel,) * 3
    elif len(kernel) == 1:
        kernel = (kernel[0],) * 3

    if isinstance(strides, int):
        strides = (strides,) * 3
    elif len(strides) == 1:
        strides = (strides[0],) * 3

    if data_format == "NCDHW":
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

    res = general_pool(x, 0.0, jlax.add, kernel, strides, padding, 3)

    res = res / general_pool(
        jnp.ones_like(x, dtype=res.dtype), 0.0, jlax.add, kernel, strides, padding, 3
    )

    if data_format == "NCDHW":
        res = jnp.transpose(x, (0, 2, 3, 4, 1))

    return res


def dct(
    x: JaxArray,
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if norm not in (None, "ortho"):
        raise ValueError("Norm must be either None or 'ortho'")
    if axis < 0:
        axis = axis + len(x.shape)
    if n is not None:
        signal_len = x.shape[axis]
        if n <= signal_len:
            local_idx = [slice(None)] * len(x.shape)
            local_idx[axis] = slice(None, n)
            x = x[local_idx]
        else:
            pad_idx = [[0, 0] for _ in range(len(x.shape))]
            pad_idx[axis][1] = n - signal_len
            x = jnp.pad(x, pad_idx)
    real_zero = jnp.array(0.0, dtype=x.dtype)
    axis_dim = x.shape[axis]
    axis_dim_float = jnp.array(axis_dim, dtype=x.dtype)

    if type == 1:
        if norm:
            raise ValueError("Normalization not supported for type-I DCT")
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(-2, 0, -1)
        x = jnp.concatenate([x, x[tuple(axis_idx)]], axis=axis)
        dct_out = jnp.real(jnp.fft.rfft(x, axis=axis))
        return dct_out

    elif type == 2:
        dct_out = jax.scipy.fft.dct(x, type=2, n=n, axis=axis, norm=norm)
        return dct_out

    elif type == 3:
        scale_dims = [1] * len(x.shape)
        scale_dims[axis] = axis_dim
        scale = 2.0 * jnp.exp(
            jlax.complex(
                real_zero, jnp.arange(axis_dim_float) * math.pi * 0.5 / axis_dim_float
            )
        ).reshape(scale_dims)
        if norm == "ortho":
            n1 = jnp.sqrt(axis_dim_float)
            n2 = n1 * jnp.sqrt(0.5)
            sf = jnp.pad(jnp.expand_dims(n1, 0), (0, axis_dim - 1), constant_values=n2)
            x = x * sf.reshape(scale_dims)
        else:
            x = x * axis_dim_float

        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(None, axis_dim)
        dct_out = jnp.real(
            jnp.fft.irfft(scale * jlax.complex(x, real_zero), n=2 * axis_dim, axis=axis)
        )[tuple(axis_idx)]
        return dct_out

    elif type == 4:
        dct_2 = jax.scipy.fft.dct(x, type=2, n=2 * axis_dim, axis=axis, norm=None)
        axis_idx = [slice(None)] * len(x.shape)
        axis_idx[axis] = slice(1, None, 2)
        dct_out = dct_2[tuple(axis_idx)]
        if norm == "ortho":
            dct_out *= math.sqrt(0.5) * jlax.rsqrt(axis_dim_float)
    return dct_out


def fft(
    x: JaxArray,
    dim: int,
    /,
    *,
    norm: Optional[str] = "backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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
    return jnp.fft.fft(x, n, dim, norm)


def dropout1d(
    x: JaxArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if training:
        if data_format == "NWC":
            perm = (0, 2, 1) if len(x.shape) == 3 else (1, 0)
            x = jnp.transpose(x, perm)
        noise_shape = list(x.shape)
        noise_shape[-1] = 1
        _, rng_input = jax.random.split(RNG.key)
        mask = jax.random.bernoulli(rng_input, 1 - prob, noise_shape)
        res = jnp.where(mask, x / (1 - prob), 0)
        if data_format == "NWC":
            res = jnp.transpose(res, perm)
        return res
    else:
        return x


def dropout3d(
    x: JaxArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if training:
        is_batched = len(x.shape) == 5
        if data_format == "NCDHW":
            perm = (0, 2, 3, 4, 1) if is_batched else (1, 2, 3, 0)
            x = jnp.transpose(x, perm)
        noise_shape = list(x.shape)
        sl = slice(1, -1) if is_batched else slice(-1)
        noise_shape[sl] = [1] * 3
        _, rng_input = jax.random.split(RNG.key)
        mask = jax.random.bernoulli(rng_input, 1 - prob, noise_shape)
        res = jnp.where(mask, x / (1 - prob), 0)
        if data_format == "NCDHW":
            perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
            res = jnp.transpose(res, perm)
        return res
    else:
        return x


def ifft(
    x: JaxArray,
    dim: int,
    *,
    norm: Optional[str] = "backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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
    return jnp.fft.ifft(x, n, dim, norm)


def interpolate(
    x: JaxArray,
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Union[Literal["linear", "bilinear"]] = "linear",
    align_corners: Optional[bool] = None,
    antialias: Optional[bool] = False,
    out: Optional[JaxArray] = None,
):
    if align_corners or mode in ["area", "nearest"]:
        return ivy.functional.experimental.interpolate(
            x, size, mode=mode, align_corners=align_corners, antialias=antialias
        )

    dims = len(x.shape) - 2
    size = (size,) * dims if isinstance(size, int) else size
    size = [x.shape[0], *size, x.shape[1]]
    x = jnp.transpose(x, (0, *range(2, dims + 2), 1))
    return jnp.transpose(
        jax.image.resize(x, shape=size, method=mode, antialias=antialias),
        (0, dims + 1, *range(1, dims + 1)),
    )
