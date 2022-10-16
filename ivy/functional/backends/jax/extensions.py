import logging
from typing import Optional, Union, Tuple, Callable, Literal, Sequence
from numbers import Number
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_coo_not_csr,
)
from ivy.functional.backends.jax import JaxArray
import jax.lax as jlax
import jax.numpy as jnp
import math


def is_native_sparse_array(x):
    """Jax does not support sparse arrays natively."""
    return False


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None,
):
    ivy.assertions.check_exists(
        data,
        inverse=True,
        message="data cannot be specified, Jax does not support sparse array natively",
    )
    if _is_coo_not_csr(
        coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
    logging.warning("Jax does not support sparse array natively, None is returned.")
    return None


def native_sparse_array_to_indices_values_and_shape(x):
    logging.warning(
        "Jax does not support sparse array natively, None is returned for        "
        " indices, values and shape."
    )
    return None, None, None


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def vorbis_window(
    window_length: JaxArray,
    *,
    dtype: Optional[jnp.dtype] = jnp.float32,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.array(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.lcm(x1, x2)


def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[jnp.dtype] = None,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    window_length = window_length + 1 if periodic is True else window_length
    return jnp.array(jnp.hanning(window_length), dtype=dtype)


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
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(window_shape), (
            f"padding {padding} must specify pads for same number of dims as "
            f"window_shape {window_shape}"
        )
        assert all(
            [len(x) == 2 for x in padding]
        ), f"each entry in padding {padding} must be length 2"
        padding = ((0, 0),) + padding + ((0, 0),)
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


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def pad(
    x: JaxArray,
    /,
    pad_width: Union[Sequence[Sequence[int]], JaxArray, int],
    *,
    mode: Optional[
        Union[
            Literal[
                "constant",
                "edge",
                "linear_ramp",
                "maximum",
                "mean",
                "median",
                "minimum",
                "reflect",
                "symmetric",
                "wrap",
                "empty",
            ],
            Callable,
        ]
    ] = "constant",
    stat_length: Optional[Union[Sequence[Sequence[int]], int]] = None,
    constant_values: Optional[Union[Sequence[Sequence[Number]], Number]] = 0,
    end_values: Optional[Union[Sequence[Sequence[Number]], Number]] = 0,
    reflect_type: Optional[Literal["even", "odd"]] = "even",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if mode in ["maximum", "mean", "median", "minimum"]:
        return jnp.pad(
            _flat_array_to_1_dim_array(x),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        return jnp.pad(
            _flat_array_to_1_dim_array(x),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        return jnp.pad(
            _flat_array_to_1_dim_array(x),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        return jnp.pad(
            _flat_array_to_1_dim_array(x),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        return jnp.pad(
            _flat_array_to_1_dim_array(x),
            pad_width,
            mode=mode,
        )


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


def moveaxis(
    a: JaxArray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.moveaxis(a, source, destination)


def heaviside(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.heaviside(x1, x2)


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


def flipud(
    m: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.flipud(m)
