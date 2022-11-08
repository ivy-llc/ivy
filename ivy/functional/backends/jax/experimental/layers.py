from typing import Optional, Union, Tuple
import ivy
from ivy.functional.backends.jax import JaxArray
import jax.lax as jlax
import jax.numpy as jnp


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
    res = _pool(x, 0.0, jlax.add, kernel, strides, padding)
    div_shape = res.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / _pool(jnp.ones(div_shape), 0.0, jlax.add, kernel, strides, padding)

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

    res = _pool(x, 0.0, jlax.add, kernel, strides, padding)
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / _pool(
        jnp.ones(div_shape, dtype=res.dtype), 0.0, jlax.add, kernel, strides, padding
    )
    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))
    return res
