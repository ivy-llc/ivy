# global
import math
from numbers import Number
from typing import Union, Optional, Tuple, List, Sequence, Iterable

import paddle

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_dtypes , with_unsupported_device_and_dtypes
# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape
from . import backend_version


# Array API Standard #
# -------------------#


def concat(
    xs: Union[Tuple[paddle.Tensor, ...], List[paddle.Tensor]],
    /,
    *,
    axis: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()

@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16","bfloat16")}}, backend_version
)
def expand_dims(
    x: paddle.Tensor,
    /,
    *,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype == paddle.float16:
        return paddle.unsqueeze(x.cast('float32'), axis).cast('float16')
    return paddle.unsqueeze(x, axis)


def flip(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def permute_dims(
    x: paddle.Tensor,
    /,
    axes: Tuple[int, ...],
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.transpose(x, axes)


def _reshape_fortran_paddle(x, shape):
    if len(x.shape) > 0:
        x = paddle.transpose(x, list(reversed(range(len(x.shape)))))
    return paddle.transpose(paddle.reshape(x, shape[::-1]), list(range(len(shape)))[::-1])


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def reshape(
    x: paddle.Tensor,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    order: Optional[str] = "C",
    allowzero: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if len(shape) == 0:
        out_scalar = True
        out_dtype = x.dtype
        if x.dtype in [paddle.int16, paddle.float16]:
            x = x.astype(paddle.float32)
        shape = [1]
    else:
        out_scalar = False
    if len(x.shape) == 0:
        x = paddle.reshape(x, shape=[1])
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, paddle.to_tensor(shape) != 0, x.shape)
        ]
    if copy:
        newarr = paddle.clone(x)
        if order == "F":
            ret = _reshape_fortran_paddle(newarr, shape)
            if out_scalar:
                return ret.squeeze().cast(out_dtype)

            return ret
        ret = paddle.reshape(newarr, shape)
        if out_scalar:

            return ret.squeeze().cast(out_dtype)

        return ret
    if order == "F":
        ret = _reshape_fortran_paddle(x, shape)
        if out_scalar:
            return ret.squeeze().cast(out_dtype)

        return ret
    ret = paddle.reshape(x, shape)
    if out_scalar:
        return ret.squeeze().cast(out_dtype)

    return ret


def roll(
    x: paddle.Tensor,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()

@with_unsupported_dtypes(
    {"2.4.2 and below": ("int16", "uint16", "float16")},
    backend_version,
)
def squeeze(
    x: paddle.Tensor,
    /,
    axis: Union[int, Sequence[int]],
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    if len(x.shape)==0:
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.utils.exceptions.IvyException(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    return paddle.squeeze(x, axis=axis)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def stack(
    arrays: Union[Tuple[paddle.Tensor], List[paddle.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    # The input list is converted to a tensor to promote the dtypes of the elements to the same dtype.
    # This is necessary because the stack function does not support mixed dtypes.
    dtype_list = set(map(lambda x: x.dtype, arrays))
    if len(dtype_list) == 1:
        dtype = dtype_list.pop()
    elif len(dtype_list) == 2:
        dtype = ivy.promote_types(*dtype_list)
    else:
        raise ValueError("Cannot promote more than 2 dtypes per stack.")

    arrays = paddle.to_tensor(arrays, dtype=dtype)
    if len(arrays.shape) == 1:  # handles scalar tensors
        return arrays
    if 'complex' in str(dtype):
        real_list = []
        imag_list = []
        for array in arrays:
            real_list.append(paddle.real(array))
            imag_list.append(paddle.imag(array))
        re_stacked = paddle.stack(real_list, axis=axis)
        imag_stacked = paddle.stack(imag_list, axis=axis)
        return re_stacked + imag_stacked * 1j
    else:
        arrays_list = []
        for array in arrays.cast('float64'):
            arrays_list.append(array)
        return paddle.stack(arrays_list, axis=axis).cast(dtype)


# Extra #
# ------#


def split(
    x: paddle.Tensor,
    /,
    *,
    num_or_size_splits: Optional[Union[int, List[int]]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()


def repeat(
    x: paddle.Tensor,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: int = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.repeat_interleave(x, repeats)


def tile(
    x: paddle.Tensor, /, repeats: Sequence[int], *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if isinstance(repeats, paddle.Tensor):
        repeats = repeats.detach().cpu().numpy().tolist()
    return x.repeat(repeats)


def constant_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def zero_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def swapaxes(
    x: paddle.Tensor, axis0: int, axis1: int, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def clip(
    x: paddle.Tensor,
    x_min: Union[Number, paddle.Tensor],
    x_max: Union[Number, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def unstack(
    x: paddle.Tensor, /, *, axis: int = 0, keepdims: bool = False
) -> List[paddle.Tensor]:
    raise IvyNotImplementedException()
