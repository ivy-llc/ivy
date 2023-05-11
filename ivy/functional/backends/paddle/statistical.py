# global

torch_scatter = None
from typing import Union, Optional, Sequence


import paddle

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version
from ivy.func_wrapper import with_unsupported_device_and_dtypes

# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def min(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            real = paddle.amin(x.real(), axis=axis, keepdim=keepdims)
            masked_x = ivy.to_native(ivy.greater_equal(x, paddle.amin(x.real())) * x)
            imag = paddle.amin(masked_x.imag(), axis=axis, keepdim=keepdims)
            return real + 1j * imag
        return paddle.amin(x.cast("float32"), axis=axis, keepdim=keepdims).cast(x.dtype)
    return paddle.amin(x, axis=axis, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            real = paddle.amax(x.real(), axis=axis, keepdim=keepdims)
            masked_x = ivy.to_native(ivy.greater_equal(x, paddle.amax(x.real())) * x)
            imag = paddle.amax(masked_x.imag(), axis=axis, keepdim=keepdims)
            return real + 1j * imag
        return paddle.amax(x.cast("float32"), axis=axis, keepdim=keepdims).cast(x.dtype)
    return paddle.amax(x, axis=axis, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def mean(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
    ]:
        if paddle.is_complex(x):
            ret = paddle.mean(x.real(), axis=axis, keepdim=keepdims) + 1j * paddle.mean(
                x.imag(), axis=axis, keepdim=keepdims
            )
            if ret.ndim == 1 and not keepdims and axis is None:
                ret = ret.squeeze()
            return ret
        ret = paddle.mean(x.cast("float32"), axis=axis, keepdim=keepdims)
        if ret.ndim == 1 and not keepdims and axis is None:
            ret = ret.squeeze()
        return ret.astype(x.dtype)
    ret = paddle.mean(x, axis=axis, keepdim=keepdims)
    if ret.ndim == 1 and not keepdims and axis is None:
        ret = ret.squeeze()
    return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def prod(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
    # TODO:prod causes segmentation fault
    return paddle.prod(x, axis=axis, keepdim=keepdims, dtype=dtype)


def _std(x, axis, correction, keepdim):
    with ivy.ArrayMode(False):
        u = mean(x, axis=axis, keepdims=True)
        out = sum(ivy.pow(ivy.subtract(x, u), 2), axis=axis, keepdims=keepdim)
        num_elm_in = paddle.prod(paddle.to_tensor(x.shape)).item()
        num_elm_out = paddle.prod(paddle.to_tensor(out.shape)).item()
        n = num_elm_out / num_elm_in
        out = ivy.sqrt(ivy.multiply(out, n))
        if correction:
            n = ivy.sqrt(
                ivy.divide(num_elm_in, (num_elm_in - correction * num_elm_out))
            )
            out = ivy.multiply(out, n)
        return out


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def std(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return _std(x, axis, correction, keepdims).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def sum(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[paddle.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = x.dtype if dtype is None else dtype
    dtype = ivy.as_ivy_dtype(dtype)
    if x.dtype in [paddle.int8, paddle.uint8]:
        return paddle.sum(x.cast("float32"), axis=axis, dtype=dtype, keepdim=keepdims)
    return paddle.sum(x, axis=axis, dtype=dtype, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def var(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        ret = ivy.pow(_std(x, axis, correction, keepdims), 2).cast(x.dtype)
    return ret


# Extra #
# ----- #
@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "uint8", "int16")}},
    backend_version,
)
def cumprod(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = dtype if dtype is not None else x.dtype
    if dtype in [paddle.uint8, paddle.int8, paddle.int16]:
        x = paddle.cast(x, "int32")
    else:
        x = paddle.cast(x, dtype)
    if not (exclusive or reverse):
        return paddle.cumprod(x, dim=axis).cast(dtype)
    elif exclusive and reverse:
        with ivy.ArrayMode(False):
            x = paddle.cumprod(ivy.flip(x, axis=(axis,)), dim=axis)
            x = ivy.swapaxes(x, axis, -1)
            x = ivy.concat((ivy.ones_like(x[..., -1:]), x[..., :-1]), axis=-1)
            x = ivy.swapaxes(x, axis, -1)
            return ivy.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        with ivy.ArrayMode(False):
            x = ivy.swapaxes(x, axis, -1)
            x = ivy.concat((ivy.ones_like(x[..., -1:]), x[..., :-1]), axis=-1)
            x = paddle.cumprod(x, -1)
            return ivy.swapaxes(x, axis, -1).cast(dtype)
    else:
        with ivy.ArrayMode(False):
            x = paddle.cumprod(ivy.flip(x, axis=(axis,)), dim=axis)
            return ivy.flip(x, axis=axis).cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "uint8", "int8", "int16")}},
    backend_version,
)
def cummax(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = dtype if dtype is not None else x.dtype
    if reverse:
        x = paddle.flip(x, axis=[axis])
    x_unstacked = paddle.unbind(x, axis=axis)
    cummax_x_unstacked = []
    cummax_x_unstacked.append(x_unstacked[0])
    for i, x_sub in enumerate(x_unstacked[1:]):
        cummax_x_sub = paddle.maximum(cummax_x_unstacked[i], x_sub)
        cummax_x_unstacked.append(cummax_x_sub)
    cummax_x = paddle.stack(cummax_x_unstacked, axis=axis)
    if reverse:
        cummax_x = paddle.flip(cummax_x, axis=[axis])
    return cummax_x.cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "uint8", "int8", "int16")}},
    backend_version,
)
def cummin(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = dtype if dtype is not None else x.dtype
    if reverse:
        x = paddle.flip(x, axis=[axis])
    x_unstacked = paddle.unbind(x, axis=axis)
    cummin_x_unstacked = []
    cummin_x_unstacked.append(x_unstacked[0])
    for i, x_sub in enumerate(x_unstacked[1:]):
        cummin_x_sub = paddle.minimum(cummin_x_unstacked[i], x_sub)
        cummin_x_unstacked.append(cummin_x_sub)
    cummin_x = paddle.stack(cummin_x_unstacked, axis=axis)
    if reverse:
        cummin_x = paddle.flip(cummin_x, axis=[axis])
    return cummin_x.cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "complex64", "complex128")}},
    backend_version,
)
def cumsum(
    x: paddle.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = dtype if dtype is not None else x.dtype
    if ivy.as_native_dtype(dtype) in [
        paddle.uint8,
        paddle.int8,
        paddle.float16,
        paddle.bool,
    ]:
        x = paddle.cast(x, "float32")
    else:
        x = paddle.cast(x, dtype)
    if not (exclusive or reverse):
        return paddle.cumsum(x, axis=axis).cast(dtype)
    elif exclusive and reverse:
        with ivy.ArrayMode(False):
            x = paddle.cumsum(ivy.flip(x, axis=(axis,)), axis=axis)
            x = ivy.swapaxes(x, axis, -1)
            x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
            x = ivy.swapaxes(x, axis, -1)
            return ivy.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        with ivy.ArrayMode(False):
            x = ivy.swapaxes(x, axis, -1)
            x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
            x = paddle.cumsum(x, -1)
            return ivy.swapaxes(x, axis, -1).cast(dtype)
    else:
        with ivy.ArrayMode(False):
            x = paddle.cumsum(ivy.flip(x, axis=(axis,)), axis=axis)
            return ivy.flip(x, axis=axis).cast(dtype)


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
