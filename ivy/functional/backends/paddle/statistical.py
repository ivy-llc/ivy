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
            if x.ndim == 1 and not keepdims:
                ret = ret.squeeze()
            return ret
        ret = paddle.mean(x.cast("float32"), axis=axis, keepdim=keepdims)
        if x.ndim == 1 and not keepdims:
            ret = ret.squeeze()
        return ret.astype(x.dtype)
    ret = paddle.mean(x, axis=axis, keepdim=keepdims)
    if x.ndim == 1 and not keepdims:
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
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.uint8]:
        dtype = x.dtype if dtype is None else dtype
        return paddle.sum(
            x.cast("float32"), axis=axis, dtype=dtype, keepdim=keepdims
        ).cast(dtype)
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
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
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


def cumsum(
    x: paddle.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is None:
        if dtype is paddle.bool:
            dtype = paddle.framework.get_default_dtype()
        elif paddle.framework.in_dygraph_mode() and x.dtype in [
            paddle.bool,
            paddle.uint8,
            paddle.int8,
            paddle.int16,
        ]:
            dtype = paddle.int32
        elif x.dtype in [paddle.uint8, paddle.int8, paddle.int16]:
            dtype = paddle.framework.get_default_dtype()
        else:
            dtype = x.dtype
    if exclusive or reverse:
        if exclusive and reverse:
            x = paddle.flip(x, [axis])
            x = paddle.cumsum(x, axis=axis, dtype=dtype)
            x = paddle.concat([paddle.zeros_like(x[..., -1:]), x[..., :-1]], axis=-1)
            res = paddle.flip(x, [axis])
        elif exclusive:
            x = paddle.concat([paddle.zeros_like(x[..., -1:]), x[..., :-1]], axis=-1)
            x = paddle.cumsum(x, axis=-1, dtype=dtype)
            res = x
        elif reverse:
            x = paddle.flip(x, [axis])
            x = paddle.cumsum(x, axis=axis, dtype=dtype)
            res = paddle.flip(x, [axis])
        return res
    return paddle.cumsum(x, axis=axis, dtype=dtype)


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.einsum(equation, *operands, out=out)
