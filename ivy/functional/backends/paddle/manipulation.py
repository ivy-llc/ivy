# global
from numbers import Number
from typing import Union, Optional, Tuple, List, Sequence, Iterable
import math
import paddle

# local
import ivy
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import (
    with_unsupported_device_and_dtypes,
    with_unsupported_dtypes,
    with_supported_dtypes,
)

# noinspection PyProtectedMember
from . import backend_version
from ...ivy.manipulation import _calculate_out_shape


# Array API Standard #
# -------------------#


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "float16")},
    backend_version,
)
def concat(
    xs: Union[Tuple[paddle.Tensor, ...], List[paddle.Tensor]],
    /,
    *,
    axis: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtypes_list = list(set(map(lambda x: x.dtype, xs)))
    dtype = dtypes_list.pop()
    if len(dtypes_list) > 0:
        for d in dtypes_list:
            dtype = ivy.promote_types(dtype, d)
    xs = list(map(lambda x: x.cast("int32" if dtype == paddle.int16 else dtype), xs))
    if all(0 in x.shape for x in xs):
        shapes = [x.shape for x in xs]
        if any(len(s) != len(shapes[0]) for s in shapes):
            raise ivy.exceptions.IvyValueError(
                "all the input arrays must have the same number of dimensions"
            )
        axis = axis + len(xs[0].shape) if axis < 0 else axis
        sizes = [[v for i, v in enumerate(s) if i != axis] for s in shapes]
        if any(s != sizes[0] for s in sizes):
            raise ivy.exceptions.IvyValueError(
                "the input arrays must have the same size along the specified axis"
            )
        ret = paddle.empty(
            [*shapes[0][:axis], sum(s[axis] for s in shapes), *shapes[0][axis + 1 :]],
            dtype=dtype,
        )
    else:
        ret = paddle.concat(xs, axis=axis)
    if dtype == paddle.int16:
        ret = ret.cast("int16")
    return ret


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float64",
            "complex128",
            "float32",
            "complex64",
            "bool",
        )
    },
    backend_version,
)
def expand_dims(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[int, Sequence[int]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    out_shape = _calculate_out_shape(axis, x.shape)
    if 0 in x.shape:
        return paddle.empty(out_shape, dtype=x.dtype)
    # reshape since unsqueeze sets a maximum limit of dimensions
    return x.reshape(out_shape)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "float16", "int16", "int8", "uint8")},
    backend_version,
)
def flip(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if axis is None:
        axis = list(range(x.ndim))
    return paddle.flip(x, axis)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("uint8", "int8", "int16", "bfloat16", "float16")},
    backend_version,
)
def permute_dims(
    x: paddle.Tensor,
    /,
    axes: Tuple[int, ...],
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if copy:
        newarr = paddle.clone(x)
        return paddle.transpose(newarr, axes)
    return paddle.transpose(x, axes)


def _reshape_fortran_paddle(x, shape):
    if len(x.shape) > 0:
        x = paddle_backend.permute_dims(x, list(reversed(range(x.ndim))))
    return paddle_backend.permute_dims(
        paddle.reshape(x, shape[::-1]), list(range(len(shape)))[::-1]
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
    if 0 in x.shape:
        if -1 in shape:
            shape = [
                (
                    s
                    if s != -1
                    else math.prod(x.shape) // math.prod([s for s in shape if s != -1])
                )
                for s in shape
            ]
        return paddle.empty(shape, dtype=x.dtype)
    if len(shape) == 0:
        out_scalar = True
        shape = [1]
    else:
        out_scalar = False
    if not allowzero:
        shape = [
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, paddle.to_tensor(shape) != 0, x.shape)
        ]
    if len(x.shape) == 0:
        x = paddle.reshape(x, shape=[1])
    if order == "F":
        ret = _reshape_fortran_paddle(x, shape)
        if out_scalar:
            return paddle_backend.squeeze(ret, axis=0)

        return ret
    ret = paddle.reshape(x, shape)
    if out_scalar:
        return paddle_backend.squeeze(ret, axis=0)

    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64", "int32", "int64")},
    backend_version,
)
def roll(
    x: paddle.Tensor,
    /,
    shift: Union[int, Sequence[int]],
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.roll(x, shift, axis)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "float16", "int16")}, backend_version
)
def squeeze(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    if len(x.shape) == 0:
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.utils.exceptions.IvyException(
            f"tried to squeeze a zero-dimensional input by axis {axis}"
        )
    if x.ndim > 6:
        # Paddle squeeze sets a maximum limit of 6 dims in the input
        x_shape = x.shape
        x_shape.pop(axis)
        return paddle_backend.reshape(x, x_shape)
    return paddle.squeeze(x, axis=axis)


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("int16", "uint8", "int8", "float16")}},
    backend_version,
)
def stack(
    arrays: Union[Tuple[paddle.Tensor], List[paddle.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype_list = set(map(lambda x: x.dtype, arrays))
    dtype = dtype_list.pop()
    if len(dtype_list) > 0:
        for d in dtype_list:
            dtype = ivy.promote_types(dtype, d)

    arrays = list(map(lambda x: x.cast(dtype), arrays))

    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        raise ValueError("Shapes of all inputs must match")
    if 0 in first_shape:
        return ivy.empty(
            first_shape[:axis] + [len(arrays)] + first_shape[axis:], dtype=dtype
        )

    if dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16, paddle.bool]:
        arrays = list(map(lambda x: x.cast("float32"), arrays))
        return paddle.stack(arrays, axis=axis).cast(dtype)

    elif dtype in [
        paddle.complex64,
        paddle.complex128,
    ]:
        arrays = list(map(lambda x: x.cast(dtype), arrays))
        real_list = list(map(lambda x: x.real(), arrays))
        imag_list = list(map(lambda x: x.imag(), arrays))
        re_stacked = paddle.stack(real_list, axis=axis)
        imag_stacked = paddle.stack(imag_list, axis=axis)
        return paddle.complex(re_stacked, imag_stacked)
    else:
        return paddle.stack(arrays, axis=axis)


# Extra #
# ------#


@with_unsupported_dtypes({"2.6.0 and below": ("int16",)}, backend_version)
def split(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[int, List[int], paddle.Tensor]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> List[paddle.Tensor]:
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise ivy.utils.exceptions.IvyException(
                "input array had no shape, but num_sections specified was"
                f" {num_or_size_splits}"
            )
        return [x]
    if num_or_size_splits is None:
        num_or_size_splits = x.shape[axis]
    elif isinstance(num_or_size_splits, paddle.Tensor):
        num_or_size_splits = num_or_size_splits.cast("int32")
        num_or_size_splits = num_or_size_splits.tolist()
    elif isinstance(num_or_size_splits, int):
        num_chunks = x.shape[axis] // num_or_size_splits
        remainder = x.shape[axis] % num_or_size_splits
        if remainder != 0:
            if with_remainder:
                num_or_size_splits = [num_or_size_splits] * num_chunks + [remainder]
            else:
                raise ivy.utils.exceptions.IvyException(
                    "Split size is not compatible with input shape"
                )

    if isinstance(num_or_size_splits, (list, tuple)):
        if sum(num_or_size_splits) < x.shape[axis]:
            num_or_size_splits + type(num_or_size_splits)([-1])
        elif sum(num_or_size_splits) > x.shape[axis]:
            raise ivy.utils.exceptions.IvyException(
                "total split size is not compatible with input shape,"
                f" got {sum(num_or_size_splits)} which is more than x.shape[axis]"
            )

    if paddle.is_complex(x):
        imag_list = paddle.split(x.imag(), num_or_size_splits, axis)
        real_list = paddle.split(x.real(), num_or_size_splits, axis)
        return [paddle.complex(a, b) for a, b in zip(real_list, imag_list)]
    return paddle.split(x, num_or_size_splits, axis)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex", "float32", "float64", "int32", "int64")},
    backend_version,
)
def repeat(
    x: paddle.Tensor,
    /,
    repeats: Union[int, Iterable[int]],
    *,
    axis: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # handle the case when repeats contains 0 as paddle doesn't support it
    if (isinstance(repeats, Number) and repeats == 0) or (
        isinstance(repeats, paddle.Tensor) and repeats.size == 1 and repeats.item() == 0
    ):
        if axis is None:
            return paddle.to_tensor([], dtype=x.dtype)
        else:
            shape = x.shape
            shape[axis] = 0
            return paddle.zeros(shape=shape).cast(x.dtype)

    if isinstance(repeats, paddle.Tensor) and repeats.size == 1:
        repeats = repeats.item()

    if axis is not None:
        axis %= x.ndim
    if paddle.is_complex(x):
        return paddle.complex(
            paddle.repeat_interleave(x.real(), repeats=repeats, axis=axis),
            paddle.repeat_interleave(x.imag(), repeats=repeats, axis=axis),
        )
    return paddle.repeat_interleave(x, repeats=repeats, axis=axis)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "float16", "int16", "int8", "uint8")},
    backend_version,
)
def tile(
    x: paddle.Tensor, /, repeats: Sequence[int], *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.ndim >= 7:
        repeats = (
            repeats.numpy().tolist() if isinstance(repeats, paddle.Tensor) else repeats
        )
        new_shape = [*x.shape[:5], -1]
        reshaped_tensor = paddle.reshape(x, new_shape)
        new_repeats = repeats[:5] + [math.prod(repeats[5:])]
        tiled_reshaped_tensor = tile(reshaped_tensor, new_repeats)
        tiled_shape = tuple(s * r for s, r in zip(x.shape, repeats))
        result = paddle.reshape(tiled_reshaped_tensor, tiled_shape)
        return result
    if ivy.min(repeats) == 0:
        # This logic is to mimic other backends behaviour when a 0 in repeat
        # is received since paddle doesn't natively support it
        if len(repeats) < x.ndim:
            shape = x.shape
            shape[-len(repeats) :] = paddle_backend.multiply(
                shape[-len(repeats) :], repeats
            ).tolist()
        elif len(repeats) > x.ndim:
            shape = (
                repeats.tolist()
                if isinstance(repeats, paddle.Tensor)
                else list(repeats)
            )
            shape[-x.ndim - 1 :] = paddle_backend.multiply(
                shape[-x.ndim - 1 :], repeats
            ).tolist()
        else:
            shape = paddle_backend.multiply(x.shape, repeats).tolist()
        return paddle.zeros(shape).cast(x.dtype)

    return paddle.tile(x, repeats)


@with_unsupported_dtypes(
    {
        "2.6.0 and below": (
            "bfloat16",
            "float16",
            "int8",
            "int16",
            "uint8",
        )
    },
    backend_version,
)
def constant_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    value: Number = 0.0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    paddings = []
    pad_width = list(pad_width)
    for item in pad_width:
        if len(item) != 2:
            raise ivy.utils.exceptions.IvyException("Length of each item should be 2")
        else:
            paddings.append(item[0])
            paddings.append(item[1])
    return paddle.nn.functional.pad(x=x, pad=paddings, value=value)


@with_unsupported_dtypes({"2.6.0 and below": ("float16",)}, backend_version)
def zero_pad(
    x: paddle.Tensor,
    /,
    pad_width: List[List[int]],
    *,
    out: Optional[paddle.Tensor] = None,
):
    return paddle_backend.constant_pad(x, pad_width=pad_width, value=0)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "bool",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    backend_version,
)
def pad_sequence(
    sequences: Union[paddle.Tensor, Iterable[Tuple[int]]],
    batch_first: bool = False,
    padding_value: Union[Iterable[Tuple[Number]], Number] = 0,
):
    raise ivy.exceptions.IvyNotImplementedException(
        "pad_sequence not implemented for Paddle backend"
    )


def swapaxes(
    x: paddle.Tensor,
    axis0: int,
    axis1: int,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    axes = [x for x in range(x.ndim)]
    axes[axis0], axes[axis1] = axes[axis1], axes[axis0]
    return paddle_backend.permute_dims(x, axes)


@with_supported_dtypes(
    {"2.6.0 and below": ("int32", "int64", "float32", "float64")},
    backend_version,
)
def clip(
    x: paddle.Tensor,
    /,
    x_min: Optional[Union[Number, paddle.Tensor]] = None,
    x_max: Optional[Union[Number, paddle.Tensor]] = None,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x_min is None and x_max is None:
        raise ValueError("At least one of the x_min or x_max must be provided")
    promoted_type = x.dtype
    if x_min is not None:
        if not hasattr(x_min, "dtype"):
            x_min = ivy.array(x_min).data
        promoted_type = ivy.as_native_dtype(ivy.promote_types(x.dtype, x_min.dtype))
        x = paddle_backend.maximum(
            paddle.cast(x, promoted_type), paddle.cast(x_min, promoted_type)
        )
    if x_max is not None:
        if not hasattr(x_max, "dtype"):
            x_max = ivy.array(x_max).data
        promoted_type = ivy.as_native_dtype(
            ivy.promote_types(promoted_type, x_max.dtype)
        )
        x = paddle_backend.minimum(
            paddle.cast(x, promoted_type), paddle.cast(x_max, promoted_type)
        )
    return x


@with_unsupported_dtypes(
    {"2.6.0 and below": ("int16", "int8", "uint8", "bfloat16")}, backend_version
)
def unstack(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[paddle.Tensor]:
    if x.ndim == 0:
        return [x]
    if axis is not None:
        axis %= x.ndim
    else:
        axis = 0
    if paddle.is_complex(x):
        real_list = paddle.unbind(x.real(), axis)
        imag_list = paddle.unbind(x.imag(), axis)
        ret = [paddle.complex(a, b) for a, b in zip(real_list, imag_list)]
    else:
        ret = paddle.unbind(x, axis)
    if keepdims:
        return [paddle_backend.expand_dims(r, axis=axis) for r in ret]
    return ret
