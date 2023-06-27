# global

torch_scatter = None

from typing import Union, Optional, Sequence, Tuple

import paddle
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_device_and_dtypes
import ivy.functional.backends.paddle as paddle_backend

# local
from . import backend_version

# Array API Standard #
# -------------------#


def min(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
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
            masked_x = paddle_backend.greater_equal(x, paddle.amin(x.real())) * x
            imag = paddle.amin(masked_x.imag(), axis=axis, keepdim=keepdims)
            ret = paddle.complex(real, imag)
        else:
            ret = paddle.amin(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.amin(x, axis=axis, keepdim=keepdims)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def max(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
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
            masked_x = paddle_backend.greater_equal(x, paddle.amax(x.real())) * x
            imag = paddle.amax(masked_x.imag(), axis=axis, keepdim=keepdims)
            ret = paddle.complex(real, imag)
        else:
            ret = paddle.amax(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.amax(x, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def mean(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = x.dtype
    if x.dtype not in [
        paddle.float32,
        paddle.float64,
    ]:
        if paddle.is_complex(x):
            ret = paddle.complex(
                paddle.mean(x.real(), axis=axis, keepdim=keepdims),
                paddle.mean(x.imag(), axis=axis, keepdim=keepdims),
            )
        else:
            ret = paddle.mean(x.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.mean(x, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == x.ndim:
            axis = None
    if (x.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


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
        ret = paddle.sum(x.cast("float32"), axis=axis, dtype=dtype, keepdim=keepdims)
    else:
        ret = paddle.sum(x.cast(dtype), axis=axis, dtype=dtype, keepdim=keepdims)
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
@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("uint8", "int16")}},
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
    if ivy.as_native_dtype(dtype) in [paddle.uint8, paddle.int8, paddle.int16]:
        x = paddle.cast(x, "int32")
    else:
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


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("uint8", "int8", "int16")}},
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
    {"2.5.0 and below": {"cpu": ("complex64", "complex128")}},
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
        x = paddle.cast(x, "int32")
    else:
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


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def cummax(
    x: paddle.Tensor,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    out: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if x.dtype in (paddle.bool, paddle.float16):
        x = paddle.cast(x, "float64")
    elif x.dtype in (paddle.int16, paddle.int8, paddle.uint8):
        x = paddle.cast(x, "int64")
    elif x.dtype in (paddle.complex128, paddle.complex64):
        x = paddle.cast(paddle.real(x), "float64")

    if not (exclusive or reverse):
        return __find_cummax(x, axis=axis)

    elif exclusive and reverse:
        x, indices = __find_cummax(ivy.flip(x, axis=(axis,)), axis=axis)
        x, indices = ivy.swapaxes(x, axis, -1), ivy.swapaxes(indices, axis, -1)
        x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
        indices = ivy.concat(
            (ivy.zeros_like(indices[..., -1:]), indices[..., :-1]), axis=-1
        )
        x, indices = ivy.swapaxes(x, axis, -1), ivy.swapaxes(indices, axis, -1)
        return ivy.flip(x, axis=(axis,)), ivy.flip(indices, axis=(axis,))

    elif exclusive:
        x = ivy.swapaxes(x, axis, -1)
        x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
        x = ivy.swapaxes(x, axis, -1)
        x, indices = __find_cummax(x, axis=axis)

        return x, indices

    else:
        x, indices = __find_cummax(ivy.flip(x, axis=(axis,)), axis=axis)
        return ivy.flip(x, axis=axis), ivy.flip(indices, axis=axis)


def __find_cummax(
    x: paddle.Tensor, axis: int = 0, dtype: Optional[paddle.dtype] = None
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    indices = []
    values = []
    x_dtype = x.dtype if dtype is None else dtype
    if (
        isinstance(x.tolist()[0], list)
        and len(x[0].shape) >= 1
        and (
            (type(x[0]) == paddle.Tensor)
            or (type(x[0]) == ivy.data_classes.array.array.Array)
        )
    ):
        if axis >= 1:
            if not isinstance(x, list):
                x = x.tolist()
            for ret1 in x:
                value, indice = __find_cummax(
                    paddle.to_tensor(ret1, dtype=x_dtype), axis=axis - 1, dtype=x_dtype
                )
                indices.append(indice)
                values.append(value)
        else:
            x_list = x.numpy()
            z_list = __get_index(x_list.tolist())
            indices, values, n1 = x_list.copy(), x_list.copy(), {}
            indices.fill(0)
            values.fill(0)
            z_list = sorted(z_list, key=lambda i: i[1])
            for y, y_index in z_list:
                multi_index = y_index
                if tuple(multi_index[1:]) not in n1:
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                    values[y_index] = y
                elif (
                    y
                    >= x_list[
                        tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))
                    ]
                ):
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                    values[y_index] = y
                else:
                    indices[y_index] = n1[tuple(multi_index[1:])]
                    values[y_index] = x_list[
                        tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))
                    ]
    else:
        if not isinstance(x, list):
            x = x.tolist()
        n = 0
        for idx, y in enumerate(x):
            if x[n] > y:
                values.append(x[n])
            elif x[n] <= y or idx == 0:
                n = idx
                values.append(y)
            indices.append(n)

    if type(x) == paddle.Tensor:
        return paddle.to_tensor(values, dtype=x.dtype), paddle.to_tensor(
            indices, dtype="int64"
        )
    else:
        return ivy.array(values, dtype=x_dtype), ivy.array(indices, dtype="int64")


def __get_index(lst, indices=None, prefix=None):
    if indices is None:
        indices = []
    if prefix is None:
        prefix = []

    if isinstance(lst, list):
        for i, sub_lst in enumerate(lst):
            sub_indices = prefix + [i]
            __get_index(sub_lst, indices, sub_indices)
    else:
        indices.append((lst, tuple(prefix)))
    return indices


def einsum(
    equation: str,
    *operands: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
