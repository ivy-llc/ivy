# global

torch_scatter = None

from typing import Union, Optional, Sequence

import paddle
import ivy
from ivy.func_wrapper import (
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
import ivy.functional.backends.paddle as paddle_backend
from ivy.utils.einsum_parser import legalise_einsum_expr

# local
from . import backend_version

# Array API Standard #
# -------------------#


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64", "int32", "int64")},
    backend_version,
)
def min(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype

    def axis_condition(axis):
        if axis is None:
            return False, False
        else:
            axis_ = axis
            if not isinstance(axis, Sequence):
                axis_ = [axis]
            if paddle.is_complex(x):
                condition_complex_imag = any([x.imag().shape[dim] > 1 for dim in axis_])
                condition_complex_real = any([x.real().shape[dim] > 1 for dim in axis_])
                return condition_complex_real, condition_complex_imag
            else:
                condition_real = any([x.shape[dim] > 1 for dim in axis_])
                return condition_real, True

    if paddle.is_complex(x):
        real = (
            paddle.amin(x.real(), axis=axis, keepdim=keepdims)
            if axis_condition(axis)[0]
            else paddle.min(x.real(), axis=axis, keepdim=keepdims)
        )
        imag = (
            paddle.amin(x.imag(), axis=axis, keepdim=keepdims)
            if axis_condition(axis)[1]
            else paddle.min(x.imag(), axis=axis, keepdim=keepdims)
        )
        ret = paddle.complex(real, imag)
    else:
        if where is not None:
            max_val = (
                ivy.iinfo(x.dtype).max
                if ivy.is_int_dtype(x.dtype)
                else ivy.finfo(x.dtype).max
            )
            max_val = max_val / 10
            # max_val becomes negative after multiplying with paddle.ones_like(x)
            # therefore reduced it
            val = paddle.ones_like(x) * max_val
            val = val.astype(ret_dtype)
            x = paddle.where(where, x, val)
        ret = (
            paddle.amin(x, axis=axis, keepdim=keepdims)
            if axis_condition(axis)[0]
            else paddle.min(x, axis=axis, keepdim=keepdims)
        )
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    if initial is not None:
        initial = paddle.to_tensor(initial, dtype=ret_dtype)
        ret = paddle.minimum(ret, initial)
    return ret.astype(ret_dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64", "int32", "int64")},
    backend_version,
)
def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype

    def axis_condition(axis):
        if axis is None:
            return False, False
        else:
            axis_ = axis
            if not isinstance(axis, Sequence):
                axis_ = [axis]
            if paddle.is_complex(x):
                condition_complex_imag = any([x.imag().shape[dim] > 1 for dim in axis_])
                condition_complex_real = any([x.real().shape[dim] > 1 for dim in axis_])
                return condition_complex_real, condition_complex_imag
            else:
                condition_real = any([x.shape[dim] > 1 for dim in axis_])
                return condition_real, True

    if paddle.is_complex(x):
        const = paddle.to_tensor(1j, dtype=x.dtype)
        real_max = (
            paddle.amax(x.real(), axis=axis, keepdim=keepdims)
            if axis_condition(axis)[0]
            else paddle.max(x.real(), axis=axis, keepdim=keepdims)
        )
        imag = paddle.where(
            x.real() == real_max, x.imag(), paddle.full_like(x.imag(), -1e10)
        )
        # we consider the number with the biggest real and imag part
        img_max = (
            paddle.amax(imag, axis=axis, keepdim=keepdims)
            if axis_condition(axis)[1]
            else paddle.max(x.real(), axis=axis, keepdim=keepdims)
        )
        img_max = paddle.cast(img_max, x.dtype)
        return paddle.add(
            paddle.cast(real_max, x.dtype), paddle.multiply(img_max, const)
        )
    else:
        ret = (
            paddle.amax(x, axis=axis, keepdim=keepdims)
            if axis_condition(axis)[0]
            else paddle.max(x, axis=axis, keepdim=keepdims)
        )

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "complex", "float32", "float64")}, backend_version
)
def mean(
    x: paddle.Tensor,
    /,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    *,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is not None:
        x = ivy.astype(x, dtype).to_native()
    if paddle.is_complex(x):
        ret = paddle.complex(
            paddle.mean(x.real(), axis=axis, keepdim=keepdims),
            paddle.mean(x.imag(), axis=axis, keepdim=keepdims),
        )
    else:
        ret = paddle.mean(x, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
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
    ret = paddle.prod(x, axis=axis, keepdim=keepdims, dtype=dtype)
    if ret.dtype != dtype:
        ret = ret.cast(dtype)
    return ret


def _std(x, axis, correction, keepdim):
    u = paddle_backend.mean(x, axis=axis, keepdims=True)
    out = paddle_backend.sum(
        paddle_backend.pow(paddle_backend.subtract(x, u), 2),
        axis=axis,
        keepdims=keepdim,
    )
    num_elm_in = paddle.prod(paddle.to_tensor(x.shape)).item()
    num_elm_out = paddle.prod(paddle.to_tensor(out.shape)).item()
    n = num_elm_out / num_elm_in
    out = paddle_backend.sqrt(paddle_backend.multiply(out, n))
    if correction:
        n = paddle_backend.sqrt(
            paddle_backend.divide(num_elm_in, (num_elm_in - correction * num_elm_out))
        )
        out = paddle_backend.multiply(out, n)
    return out


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


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "float16", "float32", "float64", "int32", "int64")},
    backend_version,
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
    ret = paddle.sum(x, axis=axis, dtype=dtype, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = paddle_backend.squeeze(ret, axis=-1)
    return ret


def var(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret = paddle_backend.pow(_std(x, axis, correction, keepdims), 2).cast(x.dtype)
    return ret


# Extra #
# ----- #
@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64", "int32", "int64")},
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
    x = paddle.cast(x, dtype)
    if not (exclusive or reverse):
        return paddle.cumprod(x, dim=axis).cast(dtype)
    elif exclusive and reverse:
        x = paddle.cumprod(paddle_backend.flip(x, axis=(axis,)), dim=axis)
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.ones_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle_backend.swapaxes(x, axis, -1)
        return paddle_backend.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.ones_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle.cumprod(x, -1)
        return paddle_backend.swapaxes(x, axis, -1).cast(dtype)
    else:
        x = paddle.cumprod(paddle_backend.flip(x, axis=(axis,)), dim=axis)
        return paddle_backend.flip(x, axis=axis).cast(dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
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
    x = paddle.cast(x, dtype)
    if not (exclusive or reverse):
        return paddle.cumsum(x, axis=axis).cast(dtype)
    elif exclusive and reverse:
        x = paddle.cumsum(paddle_backend.flip(x, axis=(axis,)), axis=axis)
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.zeros_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle_backend.swapaxes(x, axis, -1)
        return paddle_backend.flip(x, axis=(axis,)).cast(dtype)
    elif exclusive:
        x = paddle_backend.swapaxes(x, axis, -1)
        x = paddle_backend.concat(
            [
                paddle.zeros_like(
                    paddle_backend.get_item(x, (..., slice(-1, None, None)))
                ),
                paddle_backend.get_item(x, (..., slice(None, -1, None))),
            ],
            axis=-1,
        )
        x = paddle.cumsum(x, -1)
        return paddle_backend.swapaxes(x, axis, -1).cast(dtype)
    else:
        x = paddle.cumsum(paddle_backend.flip(x, axis=(axis,)), axis=axis)
        return paddle_backend.flip(x, axis=axis).cast(dtype)


@with_supported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("float32", "float64", "complex64", "complex128"),
            "gpu": (
                "bfloat16",
                "float16",
                "float32",
                "float64",
                "complex64",
                "complex128",
            ),
        },
        "2.4.2 and below": {
            "cpu": ("float32", "float64", "complex64", "complex128"),
            "gpu": ("float16", "float32", "float64", "complex64", "complex128"),
        },
    },
    backend_version,
)
def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    equation = legalise_einsum_expr(*[equation, *operands])

    dtype_list = set(map(lambda x: x.dtype, operands))
    dtype = dtype_list.pop()
    if len(dtype_list) > 0:
        for d in dtype_list:
            dtype = ivy.promote_types(dtype, d)
        operands = list(
            map(lambda x: x.cast(dtype) if x.dtype != dtype else x, operands)
        )

    return paddle.einsum(equation, *operands)
