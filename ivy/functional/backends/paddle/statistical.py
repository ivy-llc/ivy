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


def swap_axes(x, axis1, axis2):
    n = x.ndim
    if axis1 < 0:
        axis1 = n + axis1
    if axis2 < 0:
        axis2 = n + axis2
    perm = []
    for i in range(0, n):
        perm.append(i)
    perm[axis1] = axis2
    perm[axis2] = axis1
    return paddle.transpose(x, perm)


def _infer_dtype(dtype: paddle.dtype):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype


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
        return paddle.amin(
            x.cast(ivy.default_float_dtype()), axis=axis, keepdim=keepdims
        ).cast(x.dtype)
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
        return paddle.amax(
            x.cast(ivy.default_float_dtype()), axis=axis, keepdim=keepdims
        ).cast(x.dtype)
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
            return ret if keepdims else ret.squeeze()
        ret = paddle.mean(
            x.cast(ivy.default_float_dtype()), axis=axis, keepdim=keepdims
        )
        return ret.astype(x.dtype) if keepdims else ret.squeeze().astype(x.dtype)
    ret = paddle.mean(x, axis=axis, keepdim=keepdims)
    return ret if keepdims else ret.squeeze()


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
            x.cast(ivy.default_float_dtype()), axis=axis, dtype=dtype, keepdim=keepdims
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
    {
        "2.4.2 and below": {
            "cpu": (
                "uint16",
                "bfloat16",
            )
        }
    },
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
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
        dtype = ivy.as_native_dtype(dtype)
    if dtype in [paddle.uint8, paddle.int8, paddle.int16]:
        x = paddle.cast(x, ivy.closest_valid_dtype())
    else:
        x = paddle.cast(x, dtype)
    if not (exclusive or reverse):
        ret = paddle.cumprod(x, dim=axis)
    elif exclusive and reverse:
        x = paddle.cumprod(paddle.flip(x, axis=(axis,)), dim=axis)
        x = swap_axes(x, axis, -1)
        x = paddle.concat((paddle.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = swap_axes(x, axis, -1)
        ret = paddle.flip(x, axis=(axis,))
    elif exclusive:
        x = swap_axes(x, axis, -1)
        x = paddle.concat((paddle.ones_like(x[..., -1:]), x[..., :-1]), -1)
        x = paddle.cumprod(x, -1)
        ret = swap_axes(x, axis, -1)
    else:
        x = paddle.cumprod(paddle.flip(x, axis=(axis,)), dim=axis)
        ret = paddle.flip(x, axis=axis)
    return ret.cast(dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
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
    raise IvyNotImplementedException()


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
