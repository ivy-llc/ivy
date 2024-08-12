# global
from typing import Optional, Union, Tuple, List, Literal, Sequence, Callable
import jax
import jax.lax as jlax
import jax.numpy as jnp
import math

# local
import ivy
from ivy import output_to_native_arrays
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.random import RNG
from ivy.functional.ivy.experimental.general import _correct_ivy_callable
from ivy.functional.ivy.layers import (
    _handle_padding,
    _validate_max_pool_params,
    _depth_max_pooling_helper,
)
from ivy.functional.ivy.experimental.layers import (
    _padding_ceil_mode,
    _get_size,
)
from ivy.func_wrapper import with_supported_dtypes
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ivy.functional.backends.jax.experimental.manipulation import _to_nested_tuple


def _determine_depth_max_pooling(x, kernel, strides, dims, data_format="channel_last"):
    # determine depth pooling
    _, _, depth_pooling = _depth_max_pooling_helper(
        x.shape, kernel, strides, dims=dims, data_format=data_format
    )
    if depth_pooling:
        kernel = [1, 1, 1, kernel[-1]]
        strides = [1, 1, 1, strides[-1]]
    return x, kernel, strides, depth_pooling


def _pad_str_to_list(inputs, dims, padding, strides, new_window_shape):
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
    return pad_list


def general_pool(
    inputs,
    init,
    reduce_fn,
    window_shape,
    strides,
    padding,
    dim,
    dilation=1,
    ceil_mode=False,
    count_include_pad=False,
):
    # This function assumes that param validation is already done
    window_shape = tuple(window_shape)
    strides = (1,) + strides + (1,) if len(strides) == dim else strides
    dims = (1,) + window_shape + (1,) if len(window_shape) == dim else window_shape
    if isinstance(dilation, int):
        dilation = (1,) + (dilation,) * dim + (1,)
    else:
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
        window_shape[i - 1] + (dilation[i] - 1) * (window_shape[i - 1] - 1)
        for i in range(1, len(dims) - 1)
    )
    inputs, window_shape, strides, depth_pooling = _determine_depth_max_pooling(
        inputs, window_shape, strides, dim, data_format="channel_last"
    )
    if not depth_pooling:
        # manually creating padding list
        if isinstance(padding, str):
            pad_list = _pad_str_to_list(
                inputs, dims, padding, strides, new_window_shape
            )
        else:
            if isinstance(padding, int):
                padding = [(padding,) * 2] * dim
            pad_list = [(0, 0)] + list(padding) + [(0, 0)]

        if ceil_mode:
            c = []
            for i in range(len(dims) - 2):
                pad_list[i + 1], ceil = _padding_ceil_mode(
                    inputs.shape[i + 1],
                    new_window_shape[i],
                    pad_list[i + 1],
                    strides[i + 1],
                    True,
                )
                c.append(ceil)

        if count_include_pad:
            # manually pad inputs with 0 if ceil_mode is True
            # because they're not counted in average calculation
            if ceil_mode:
                ceil = [(0, c[i]) for i in range(len(dims) - 2)]
                for i in range(len(dims) - 2):
                    pad_list[i + 1] = (
                        pad_list[i + 1][0],
                        pad_list[i + 1][1] - ceil[i][1],
                    )
                inputs = jnp.pad(inputs, pad_list, mode="constant", constant_values=1.0)
                inputs = jnp.pad(
                    inputs,
                    [(0, 0)] + ceil + [(0, 0)],
                    mode="constant",
                    constant_values=0.0,
                )
            else:
                # manually pad inputs with 1s
                # because they are counted in average calculation
                inputs = jnp.pad(inputs, pad_list, mode="constant", constant_values=1.0)
            pad_list = [(0, 0)] * len(pad_list)
    elif isinstance(padding, list) and any(
        item != 0 for sublist in padding for item in sublist
    ):
        raise NotImplementedError(
            "Nonzero explicit padding is not supported for depthwise max pooling"
        )
    else:
        pad_list = [(0, 0)] * (dim + 2)

    if not ivy.is_array(inputs):
        # if dtype is not set here, jax casts it to float64
        inputs = jnp.array(inputs, dtype=jnp.float32)
    if not ivy.is_array(init):
        init = jnp.array(init, dtype=inputs.dtype)
    promoted_type = jnp.promote_types(inputs.dtype, init.dtype)
    inputs = jnp.astype(inputs, promoted_type)
    init = jnp.astype(init, promoted_type)
    y = jlax.reduce_window(
        inputs, init, reduce_fn, dims, strides, pad_list, window_dilation=dilation
    )
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def max_pool1d(
    x: JaxArray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    dilation: Union[int, Tuple[int]] = 1,
    ceil_mode: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dims = 1
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    if data_format == "NCW":
        x = jnp.transpose(x, (0, 2, 1))
        kernel = [kernel[i] for i in [0, 2, 1]] if len(kernel) == (dims + 2) else kernel
        strides = (
            [strides[i] for i in [0, 2, 1]] if len(strides) == (dims + 2) else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    res = general_pool(
        x, -jnp.inf, jlax.max, kernel, strides, padding, dims, dilation, ceil_mode
    )

    if data_format == "NCW":
        res = jnp.transpose(res, (0, 2, 1))
    return res


def max_pool2d(
    x: JaxArray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dims = 2
    odtype = x.dtype
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )

    if data_format == "NCHW":
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = (
            [kernel[i] for i in [0, 2, 3, 1]] if len(kernel) == (dims + 2) else kernel
        )
        strides = (
            [strides[i] for i in [0, 2, 3, 1]]
            if len(strides) == (dims + 2)
            else strides
        )
        padding = (
            [padding[i] for i in [0, 2, 3, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    res = general_pool(
        x, -jnp.inf, jlax.max, kernel, strides, padding, dims, dilation, ceil_mode
    )

    if data_format == "NCHW":
        res = jnp.transpose(res, (0, 3, 1, 2))

    return jnp.astype(res, odtype)


def max_pool3d(
    x: JaxArray,
    kernel: Union[int, Tuple[int, ...]],
    strides: Union[int, Tuple[int, ...]],
    padding: Union[str, int, Tuple[int], List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    dilation: Union[int, Tuple[int, ...]] = 1,
    ceil_mode: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dims = 3
    kernel, strides, padding, dilation = _validate_max_pool_params(
        kernel, strides, padding, dilation, ceil_mode, dims, data_format
    )
    if data_format == "NCDHW":
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
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
        padding = (
            [padding[i] for i in [0, 2, 3, 4, 1]]
            if isinstance(padding, list) and len(padding) == (dims + 2)
            else padding
        )

    res = general_pool(
        x, -jnp.inf, jlax.max, kernel, strides, padding, dims, dilation, ceil_mode
    )

    if data_format == "NCDHW":
        res = jnp.transpose(res, (0, 4, 1, 2, 3))

    return res


def avg_pool1d(
    x: JaxArray,
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if data_format in ("NCW", "NCL"):
        x = jnp.transpose(x, (0, 2, 1))

    if isinstance(kernel, int):
        kernel = (kernel,)
    elif len(kernel) == 1:
        kernel = (kernel[0],)

    if isinstance(strides, int):
        strides = (strides,)
    elif len(strides) == 1:
        strides = (strides[0],)

    res = general_pool(
        x, 0.0, jlax.add, kernel, strides, padding, 1, ceil_mode=ceil_mode
    )
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    res = res / general_pool(
        jnp.ones(div_shape, dtype=res.dtype),
        0.0,
        jlax.add,
        kernel,
        strides,
        padding,
        1,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )
    if data_format in ("NCW", "NCL"):
        res = jnp.transpose(res, (0, 2, 1))
    if x.dtype == "float16":
        res = jnp.astype(res, "float16")

    return res


def avg_pool2d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
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

    res = general_pool(
        x, 0.0, jlax.add, kernel, strides, padding, 2, ceil_mode=ceil_mode
    )
    div_shape = x.shape[:-1] + (1,)
    if len(div_shape) - 2 == len(kernel):
        div_shape = (1,) + div_shape[1:]
    if divisor_override is not None:
        divisor = divisor_override
    else:
        divisor = general_pool(
            jnp.ones(div_shape, dtype=res.dtype),
            0.0,
            jlax.add,
            kernel,
            strides,
            padding,
            2,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
        )
    res = res / divisor
    if data_format == "NCHW":
        return jnp.transpose(res, (0, 3, 1, 2))
    return res


def avg_pool3d(
    x: JaxArray,
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: Union[str, int, List[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
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

    res = general_pool(
        x, 0.0, jlax.add, kernel, strides, padding, 3, ceil_mode=ceil_mode
    )

    if divisor_override is not None:
        divisor = divisor_override
    else:
        divisor = general_pool(
            jnp.ones_like(x, dtype=res.dtype),
            0.0,
            jlax.add,
            kernel,
            strides,
            padding,
            3,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
        )
    res = res / divisor

    if data_format == "NCDHW":
        res = jnp.transpose(res, (0, 4, 1, 2, 3))

    return res


@with_supported_dtypes({"0.4.24 and below": ("float32", "float64")}, backend_version)
def dct(
    x: JaxArray,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if norm not in (None, "ortho"):
        raise ValueError("Norm must be either None or 'ortho'")
    if axis < 0:
        axis += len(x.shape)
    if n is not None:
        signal_len = x.shape[axis]
        if n <= signal_len:
            local_idx = [slice(None)] * len(x.shape)
            local_idx[axis] = slice(None, n)
            x = x[tuple(local_idx)]
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


def idct(
    x: JaxArray,
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return dct(x, type=inverse_type, n=n, axis=axis, norm=norm, out=out)


def fft(
    x: JaxArray,
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
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
    if norm not in {"backward", "ortho", "forward"}:
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
        x_shape = x.shape
        is_batched = len(x_shape) == 3
        if data_format == "NCW":
            perm = (0, 2, 1) if is_batched else (1, 0)
            x = jnp.transpose(x, perm)
            x_shape = x.shape
        _, rng_input = jax.random.split(RNG.key)
        mask = jax.random.bernoulli(rng_input, 1 - prob, x_shape)
        res = jnp.where(mask, x / (1 - prob), 0)
        if data_format == "NCW":
            res = jnp.transpose(res, perm)
    else:
        res = x
    return res


def dropout2d(
    x: JaxArray,
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if training:
        x_shape = x.shape
        is_batched = len(x.shape) == 4
        if data_format == "NCHW":
            perm = (0, 2, 3, 1) if is_batched else (1, 2, 0)
            x = jnp.transpose(x, perm)
            x_shape = x.shape
        _, rng_input = jax.random.split(RNG.key)
        mask = jax.random.bernoulli(rng_input, 1 - prob, x_shape)
        res = jnp.where(mask, x / (1 - prob), 0)
        if data_format == "NCHW":
            perm = (0, 3, 1, 2) if is_batched else (2, 0, 1)
            res = jnp.transpose(res, perm)
    else:
        res = x
    return res


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
        x_shape = x.shape
        is_batched = len(x_shape) == 5
        if data_format == "NCDHW":
            perm = (0, 2, 3, 4, 1) if is_batched else (1, 2, 3, 0)
            x = jnp.transpose(x, perm)
            x_shape = x.shape
        _, rng_input = jax.random.split(RNG.key)
        mask = jax.random.bernoulli(rng_input, 1 - prob, x_shape)
        res = jnp.where(mask, x / (1 - prob), 0)
        if data_format == "NCDHW":
            perm = (0, 4, 1, 2, 3) if is_batched else (3, 0, 1, 2)
            res = jnp.transpose(res, perm)
    else:
        res = x
    return res


def ifft(
    x: JaxArray,
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
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
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return jnp.fft.ifft(x, n, dim, norm)


def interpolate(
    x: JaxArray,
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
    out: Optional[JaxArray] = None,
):
    input_size = ivy.shape(x)[2:]
    dims = len(input_size)
    size, _ = _get_size(scale_factor, size, dims, input_size)
    if all(a == b for a, b in zip(size, input_size)):
        ret = x
    else:
        mode = (
            "nearest"
            if mode == "nearest-exact"
            else "bicubic" if mode == "tf_bicubic" else mode
        )

        size = [x.shape[0], *size, x.shape[1]]
        x = jnp.transpose(x, (0, *range(2, dims + 2), 1))
        ret = jnp.transpose(
            jax.image.resize(x, shape=size, method=mode, antialias=antialias),
            (0, dims + 1, *range(1, dims + 1)),
        )
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


interpolate.partial_mixed_handler = (
    lambda *args, mode="linear", recompute_scale_factor=None, align_corners=None, **kwargs: mode  # noqa: E501
    not in [
        "area",
        "nearest",
        "nd",
        "tf_area",
        "mitchellcubic",
        "gaussian",
        "bicubic",
    ]
    and not align_corners
    and recompute_scale_factor
)


def reduce_window(
    operand: JaxArray,
    init_value: Union[int, float],
    computation: Callable,
    window_dimensions: Union[int, Sequence[int]],
    /,
    *,
    window_strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    base_dilation: Union[int, Sequence[int]] = 1,
    window_dilation: Union[int, Sequence[int]] = 1,
) -> JaxArray:
    computation = _correct_ivy_callable(computation)
    computation = output_to_native_arrays(computation)
    window_dimensions, window_strides, padding, base_dilation, window_dilation = map(
        lambda x: tuple([x] * len(operand.shape)) if isinstance(x, int) else x,
        [window_dimensions, window_strides, padding, base_dilation, window_dilation],
    )
    if not isinstance(padding, str):
        # for containers the padding reaches the function as a list of lists instead of
        # a list of tuples, which gives an unhashable dtype error
        # this is similarly a problem in the jax backend of ivy.pad
        padding = _to_nested_tuple(padding)
    return jlax.reduce_window(
        operand,
        jnp.astype(jnp.array(init_value), operand.dtype),
        computation,
        window_dimensions,
        window_strides,
        padding,
        base_dilation,
        window_dilation,
    )


def fft2(
    x: JaxArray,
    *,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions.check_elem_in_list(
        norm,
        ["backward", "ortho", "forward"],
        message=f"Unrecognized normalization mode {norm}",
    )
    if not all(isinstance(j, int) for j in dim):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {dim} to be a sequence of integers <class integer>"
        )
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
    return jnp.astype(jnp.fft.fft2(x, s, dim, norm), jnp.complex128)


def ifftn(
    x: JaxArray,
    s: Optional[Union[int, Tuple[int]]] = None,
    axes: Optional[Union[int, Tuple[int]]] = None,
    *,
    norm: str = "backward",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fft.ifftn(x, s, axes, norm)


@with_unsupported_dtypes(
    {"0.4.24 and below": ("bfloat16", "float16", "complex")}, backend_version
)
def embedding(
    weights: JaxArray,
    indices: JaxArray,
    /,
    *,
    max_norm: Optional[int] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ivy.utils.assertions.check_equal(
        len(weights.shape), 2, message="weights must be 2-d", as_array=False
    )

    embeddings = jnp.take(weights, indices, axis=0)
    if max_norm is not None:
        norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = jnp.where(
            norms > max_norm, embeddings * max_norm / norms, embeddings
        )
        embeddings = jnp.where(
            norms < -max_norm, embeddings * -max_norm / norms, embeddings
        )
    return embeddings


def rfft(
    x: JaxArray,
    /,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x = x.real
    if x.dtype == jnp.float16:
        x = jnp.astype(x, jnp.float32)

    ret = jnp.fft.rfft(x, n=n, axis=axis, norm=norm)

    if x.dtype != jnp.float64:
        ret = jnp.astype(ret, jnp.complex64)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes({"0.4.24 and below": ("float16", "complex")}, backend_version)
def rfftn(
    x: JaxArray,
    s: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
    *,
    norm: str = "backward",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if not all(isinstance(j, int) for j in axes):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {axes} to be a sequence of integers <class integer>"
        )
    if s is None:
        s = (x.shape[axes[0]], x.shape[axes[1]])
    if all(j < -len(x.shape) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {axes}, expecting ranging"
            f" from {-len(x.shape)} to {len(x.shape)-1}"
        )
    if not all(isinstance(j, int) for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting {s} to be a sequence of integers <class integer>"
        )
    if all(j <= 1 for j in s):
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {s}, expecting s points larger than 1"
        )
    if norm not in {"backward", "ortho", "forward"}:
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return jnp.astype(jnp.fft.rfftn(x, s, axes, norm), jnp.complex128)


# stft
def stft(
    signals: JaxArray,
    frame_length: int,
    frame_step: int,
    /,
    *,
    fft_length: Optional[int] = None,
    window_fn: Optional[Callable] = None,
    pad_end: Optional[bool] = False,
    name: Optional[str] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
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

    input_dtype = signals.dtype
    if input_dtype == jnp.float32:
        dtype = jnp.complex64
    elif input_dtype == jnp.float64:
        dtype = jnp.complex128

    def stft_1D(signals, frame_length, frame_step, fft_length, pad_end):
        if fft_length is None:
            fft_length = 1
            while fft_length < frame_length:
                fft_length *= 2

        num_samples = signals.shape[-1]

        if pad_end:
            num_samples = signals.shape[-1]
            num_frames = -(-num_samples // frame_step)
            pad_length = max(
                0, frame_length + frame_step * (num_frames - 1) - num_samples
            )

            signals = jnp.pad(signals, [(0, pad_length)])
        else:
            num_frames = 1 + (num_samples - frame_length) // frame_step

        stft_result = []

        if window_fn is None:
            window = 1
        else:
            window = window_fn(frame_length)

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_length
            frame = signals[..., start:end]
            windowed_frame = frame * window
            pad_length = fft_length - frame_length
            windowed_frame = jnp.pad(windowed_frame, [(0, pad_length)])
            windowed_frame = jnp.asarray(windowed_frame, dtype=dtype)

            fft_frame = jnp.fft.fft(windowed_frame, axis=-1)
            slit = int(fft_length // 2 + 1)
            stft_result.append(fft_frame[..., 0:slit])

        stft = jnp.stack(stft_result, axis=0)
        return stft

    def stft_helper(nested_list, frame_length, frame_step, fft_length):
        nested_list = nested_list
        if len(jnp.shape(nested_list)) > 1:
            return [
                stft_helper(sublist, frame_length, frame_step, fft_length)
                for sublist in nested_list
            ]
        else:
            return stft_1D(nested_list, frame_length, frame_step, fft_length, pad_end)

    to_return = stft_helper(signals, frame_length, frame_step, fft_length)
    return jnp.asarray(to_return, dtype=dtype)
