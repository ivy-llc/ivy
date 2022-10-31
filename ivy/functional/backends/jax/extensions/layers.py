from typing import Optional, Union, Tuple, Callable, Literal, Sequence, Any
from numbers import Number
import ivy
from ivy.functional.backends.jax import JaxArray
import jax
import jax.lax as jlax
import jax.numpy as jnp
import math


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


def _flat_array_to_1_dim_array(x):
    return x.reshape((1,)) if x.shape == () else x


def pad(
    input: JaxArray,
    pad_width: Union[Sequence[Sequence[int]], JaxArray, int],
    /,
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
    **kwargs: Optional[Any],
) -> JaxArray:
    if callable(mode):
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            **kwargs,
        )
    if mode in ["maximum", "mean", "median", "minimum"]:
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            stat_length=stat_length,
        )
    elif mode == "constant":
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            constant_values=constant_values,
        )
    elif mode == "linear_ramp":
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            end_values=end_values,
        )
    elif mode in ["reflect", "symmetric"]:
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
            reflect_type=reflect_type,
        )
    else:
        return jnp.pad(
            _flat_array_to_1_dim_array(input),
            pad_width,
            mode=mode,
        )


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
