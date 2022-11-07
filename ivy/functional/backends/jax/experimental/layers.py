from typing import Optional, Union, Tuple, Literal
import ivy
from ivy.functional.backends.jax import JaxArray
import jax
import jax.lax as jlax
import jax.numpy as jnp
import math


def _pool(inputs, init, reduce_fn, window_shape, strides, padding):

    if isinstance(strides, int):
        strides = (strides,) * len(window_shape)
    elif len(strides) == 1:
        strides = (strides[0],) * len(window_shape)

    assert len(window_shape) == len(
        strides
    ), f"len({window_shape}) must equal len({strides})"

    window_shape = tuple(window_shape)
    strides = (1,) + strides + (1,)
    dims = (1,) + window_shape + (1,)

    is_single_input = False
    if inputs.ndim == len(dims) - 1:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    y = jlax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def max_pool2d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if data_format == "NCHW":
        x = jnp.transpose(x, (0, 2, 3, 1))

    res = _pool(x, -jnp.inf, jlax.max, kernel, strides, padding)

    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))

    return res


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

    res = _pool(x, -jnp.inf, jlax.max, kernel, strides, padding)

    if data_format == "NCW":
        res = jnp.transpose(x, (0, 2, 1))
    return res


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if periodic is False:
        return jnp.array(jnp.kaiser(M=window_length, beta=beta), dtype=dtype)
    else:
        return jnp.array(jnp.kaiser(M=window_length + 1, beta=beta)[:-1], dtype=dtype)


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
    res = _pool(x, -jnp.inf, jlax.max, kernel, strides, padding)

    if data_format == "NCDHW":
        res = jnp.transpose(x, (0, 2, 3, 4, 1))

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

    x_shape = list(x.shape[1:4])
    pad_d = ivy.handle_padding(x_shape[0], strides[0], kernel[0], padding)
    pad_h = ivy.handle_padding(x_shape[1], strides[1], kernel[1], padding)
    pad_w = ivy.handle_padding(x_shape[2], strides[2], kernel[2], padding)

    x = jnp.pad(
        x,
        [
            (0, 0),
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
            (0, 0),
        ],
        "edge",
    )
    res = _pool(x, 0., jlax.add, kernel, strides, padding)
    div_shape = res.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / _pool(jnp.ones(div_shape), 0., jlax.add, kernel, strides, padding)

    if data_format == "NCDHW":
        res = jnp.transpose(x, (0, 2, 3, 4, 1))

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

    res = _pool(x, 0., jlax.add, kernel, strides, padding)
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / _pool(jnp.ones(div_shape,
                               dtype=res.dtype),
                      0.,
                      jlax.add,
                      kernel,
                      strides,
                      padding)
    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))
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
