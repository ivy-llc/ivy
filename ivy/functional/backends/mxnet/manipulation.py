# global
import mxnet as mx
import math
import numpy as np
from typing import Union, Tuple, Optional, List, Sequence
from ivy.functional.backends.mxnet import (
    _flat_array_to_1_dim_array,
    _handle_flat_arrays_in_out,
    _handle_flat_arrays_in,
    _1_dim_array_to_flat_array,
)

# local
import ivy

# noinspection PyProtectedMember
from ivy.functional.ivy.manipulation import _calculate_out_shape


def flip(
    x: mx.nd.NDArray,
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    num_dims = len(x.shape)
    if not num_dims:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    if axis is None:
        new_axis = list(range(num_dims))
    else:
        new_axis = axis
    if type(new_axis) is int:
        new_axis = [new_axis]
    else:
        new_axis = new_axis
    new_axis = [item + num_dims if item < 0 else item for item in new_axis]
    ret = mx.nd.flip(x, new_axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def expand_dims(
    x: mx.nd.NDArray,
    axis: Union[int, Tuple[int], List[int]] = 0,
) -> mx.nd.NDArray:
    if x.shape == ():
        ret = _flat_array_to_1_dim_array(x)
    else:
        out_shape = _calculate_out_shape(axis, x.shape)
        ret = x.reshape(out_shape)
    return ret


def stack(
    x: Union[Tuple[mx.nd.NDArray], List[mx.nd.NDArray]],
    axis: Optional[int] = 0,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    ret = mx.nd.stack(x, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def squeeze(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[mx.nd.NDArray] = None,
):
    if type(x) == ivy.Array and x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            if ivy.exists(out):
                return ivy.inplace_update(out, x)
            return x
        raise Exception(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    ret = mx.nd.squeeze(x, axis)
    if axis is None:
        ret = _1_dim_array_to_flat_array(ret)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def reshape(
    x: mx.nd.NDArray,
    shape: Union[ivy.NativeShape, Sequence[int]],
    copy: Optional[bool] = None,
) -> mx.nd.NDArray:
    if copy:
        newarr = x.copy()
        return newarr.reshape(shape)
    return x.reshape(shape)


@_handle_flat_arrays_in_out
def concat(xs, axis=-1, out: Optional[mx.nd.NDArray] = None):
    ret = mx.nd.concat(*xs, dim=axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# Extra #
# ------#


def split(x, num_or_size_splits=None, axis=0, with_remainder=False):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                "input array had no shape, but num_sections specified was {}".format(
                    num_or_size_splits
                )
            )
        return [x]
    if num_or_size_splits == 1:
        return [x]
    elif with_remainder and isinstance(num_or_size_splits, int):
        num_or_size_splits = (
            x.shape[axis] if not num_or_size_splits else num_or_size_splits
        )
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder_size = int((num_chunks - num_chunks_int) * num_or_size_splits)
        num_or_size_splits = [num_or_size_splits] * num_chunks_int + [remainder_size]
    if isinstance(num_or_size_splits, (list, tuple)):
        csum = [0] + np.cumsum(num_or_size_splits).tolist()
        starts = csum[:-1]
        ends = csum[1:]
        if axis < 0:
            slices = [
                tuple(
                    [Ellipsis, slice(s, e, 1)]
                    + [slice(None, None, None)] * int(abs(axis) - 1)
                )
                for s, e in zip(starts, ends)
            ]
        else:
            slices = [
                tuple([slice(None, None, None)] * axis + [slice(s, e, 1)])
                for s, e in zip(starts, ends)
            ]
        return [x[so] for so in slices]
    return mx.nd.split(
        x, x.shape[axis] if not num_or_size_splits else num_or_size_splits, axis
    )


@_handle_flat_arrays_in_out
def repeat(x, repeats, axis=None, out: Optional[mx.nd.NDArray] = None):
    ret = mx.nd.repeat(x, repeats, axis)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def tile(x, reps, out: Optional[mx.nd.NDArray] = None):
    if isinstance(reps, mx.nd.ndarray.NDArray):
        reps = reps.asnumpy().tolist()
    ret = mx.nd.tile(_flat_array_to_1_dim_array(x), reps)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_flat_arrays_in
def constant_pad(x, pad_width, value=0, out: Optional[mx.nd.NDArray] = None):
    if isinstance(pad_width, mx.nd.NDArray):
        pad_width = pad_width.asnumpy().tolist()
    x_shape = list(x.shape)
    num_dims = len(x_shape)
    if num_dims > 3:
        raise Exception(
            "Invalid inputs. Pad for mxnet only supports inputs with "
            "3 dimensions or smaller."
        )
    num_dims_to_add = 4 - num_dims
    new_shape = tuple([1] * num_dims_to_add + x_shape)
    mat_expanded_dims = mx.nd.reshape(x, new_shape)
    pad_width_flat = [0] * num_dims_to_add * 2 + [
        item for sublist in pad_width for item in sublist
    ]
    pad_expanded_dims = mx.nd.pad(
        mat_expanded_dims,
        mode="constant",
        pad_width=tuple(pad_width_flat),
        constant_value=value,
    )
    new_shape = [
        orig_dim + pad_width_item[0] + pad_width_item[1]
        for orig_dim, pad_width_item in zip(x_shape, pad_width)
    ]
    ret = mx.nd.reshape(pad_expanded_dims, tuple(new_shape))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def zero_pad(x, pad_width, out: Optional[mx.nd.NDArray] = None):
    ret = constant_pad(x, pad_width, 0)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


swapaxes = mx.nd.swapaxes


@_handle_flat_arrays_in_out
def clip(
    x: mx.ndarray.ndarray.NDArray,
    x_min: float,
    x_max: float,
    out: Optional[mx.ndarray.ndarray.NDArray] = None,
) -> mx.ndarray.ndarray.NDArray:
    ret = mx.nd.clip(mx.nd.array(x), x_min, x_max)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret
