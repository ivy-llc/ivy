# global
from typing import Optional, Union, Sequence
import ivy

_round = round
import mxnet as mx
import numpy as np
from numbers import Number
import multiprocessing as _multiprocessing

# local
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.mxnet.device import dev
from ivy.functional.backends.mxnet import (
    _handle_flat_arrays_in_out,
    _mxnet_init_context,
)


def is_native_array(x, exclusive=False):
    if isinstance(x, mx.nd.NDArray):
        if exclusive and x.grad is not None:
            return False
        return True
    return False


def copy_array(x: mx.nd.NDArray) -> mx.nd.NDArray:
    return x.copy()


def array_equal(x0: mx.nd.NDArray, x1: mx.nd.NDArray) -> bool:
    if ivy.dtype(x0) == "bool":
        x0 = x0.astype("int32")
    if ivy.dtype(x1) == "bool":
        x1 = x1.astype("int32")
    return mx.nd.min(mx.nd.broadcast_equal(x0, x1)) == 1


def to_numpy(x: mx.nd.NDArray, copy: bool = True) -> mx.nd.NDArray:
    if isinstance(x, np.ndarray):
        return x
    else:
        if isinstance(x, (int, float)):
            return np.array(x)
        else:
            return x.asnumpy()


def to_scalar(x: mx.nd.NDArray) -> Number:
    if isinstance(x, Number):
        return x
    else:
        x.asscalar().item()


def to_list(x: mx.nd.NDArray) -> list:
    return to_numpy(x).tolist()


@_handle_flat_arrays_in_out
def floormod(
    x: mx.nd.NDArray,
    y: mx.nd.NDArray,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    ret = x % y
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


container_types = lambda: []


def unstack(x, axis, keepdims=False):
    if x.shape == ():
        return [x]
    num_outputs = x.shape[axis]
    ret = mx.nd.split(x, num_outputs, axis, squeeze_axis=not keepdims)
    return ret if isinstance(ret, list) else [ret]


def inplace_update(
    x: Union[ivy.Array, mx.nd.NDArray],
    val: Union[ivy.Array, mx.nd.NDArray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native[:] = val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_arrays_supported():
    return True


inplace_variables_supported = lambda: True


def inplace_decrement(x, val):
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native[:] -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, mx.nd.NDArray],
    val: Union[ivy.Array, mx.nd.NDArray],
) -> Union[ivy.Array, ivy.Container]:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native[:] += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def cumsum(
    x: mx.nd.NDArray,
    axis: int = 0,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    if ivy.exists(out):
        return ivy.inplace_update(
            out, mx.nd.cumsum(x, axis if axis >= 0 else axis % len(x.shape))
        )
    else:
        mx.nd.cumsum(x, axis if axis >= 0 else axis % len(x.shape))


def cumprod(
    x: mx.nd.NDArray,
    axis: int = 0,
    exclusive: Optional[bool] = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.nd.NDArray:
    array_stack = [mx.nd.expand_dims(chunk, axis) for chunk in unstack(x, axis)]
    if exclusive:
        array_stack = [mx.nd.ones_like(array_stack[0])] + array_stack[:-1]
    new_array_list = [array_stack[0]]
    for array_chunk in array_stack[1:]:
        new_array_list.append(new_array_list[-1] * array_chunk)
    if ivy.exists(out):
        return ivy.inplace_update(out, mx.nd.concat(*new_array_list, dim=axis))
    return mx.nd.concat(*new_array_list, dim=axis)


# noinspection PyShadowingNames
def scatter_flat(
    indices, updates, size=None, tensor=None, reduction="sum", device=None
):
    if ivy.exists(tensor):
        raise Exception(
            "MXNet scatter_flat does not support scattering into "
            "an pre-existing tensor."
        )
    if reduction == "replace":
        return mx.nd.scatter_nd(updates, mx.nd.expand_dims(indices, 0), [size]).copyto(
            _mxnet_init_context(default_device(device))
        )
    else:
        raise Exception(
            "MXNet scatter_flat currently only supports reduction mode 'replace', "
            "but {} selected.".format(reduction)
        )


# noinspection PyShadowingNames
def scatter_nd(
    indices,
    updates,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    tensor=None,
    reduction="sum",
    device=None,
):
    if ivy.exists(tensor):
        raise Exception(
            "MXNet scatter_flat does not support scattering into "
            "an pre-existing tensor."
        )
    if device is None:
        device = dev(indices)
    shape = list(shape)
    num_idx_dims = len(indices.shape)
    transpose_order = [num_idx_dims - 1] + list(range(num_idx_dims - 1))
    indices = mx.nd.transpose(indices, transpose_order)
    shape = shape if type(shape) is list else shape.asnumpy().astype(np.int32).tolist()
    if reduction == "replace":
        return mx.nd.scatter_nd(updates, indices, shape).copyto(
            _mxnet_init_context(device)
        )
    else:
        raise Exception(
            "MXNet scatter_nd currently only supports reduction mode 'replace', "
            "but {} selected.".format(reduction)
        )


def gather(
    params: mx.nd.NDArray,
    indices: mx.nd.NDArray,
    axis: Optional[int] = -1,
    device: Optional[str] = None,
    out: mx.nd.NDArray = None,
) -> mx.nd.NDArray:
    if device is None:
        device = dev(params)
    index_slices = unstack(indices, -1)
    res = mx.nd.concat(
        *[
            mx.nd.expand_dims(mx.nd.pick(params, idx_slice, axis), -1)
            for idx_slice in index_slices
        ],
        dim=-1
    )
    res = mx.nd.reshape(res, indices.shape)
    if ivy.exists(out):
        out = _mxnet_init_context(device)
        return res.copyto(out)
    else:
        return res.copyto(_mxnet_init_context(device))


def gather_nd(params, indices, device=None):
    if device is None:
        device = dev(params)
    indices_shape = indices.shape
    num_idx_dims = len(indices_shape)
    transpose_order = [num_idx_dims - 1] + list(range(num_idx_dims - 1))
    indices = mx.nd.transpose(indices, transpose_order)
    return mx.nd.gather_nd(params, indices).copyto(_mxnet_init_context(device))


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def one_hot(indices: mx.nd.NDArray, depth: int, *, device: mx.context.Context):
    return mx.nd.one_hot(indices, depth)


def shape(
    x: mx.nd.NDArray, as_array: bool = False
) -> Union[mx.nd.NDArray, ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(mx.nd.shape_array(x))
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x, as_tensor=False):
    return (
        mx.nd.shape_array(mx.nd.shape_array(x)).reshape([])
        if as_tensor
        else len(x.shape)
    )


def indices_where(x):
    x_shape = x.shape
    x_flat = x.reshape(
        (
            1,
            -1,
        )
    )
    flat_indices = x_flat.astype("int32").tostype("csr").indices
    if flat_indices.shape == (0,):
        res = flat_indices.reshape((0, len(x_shape)))
        return res
    res = mx.nd.swapaxes(mx.nd.unravel_index(flat_indices, x_shape), 0, 1)
    return res


current_backend_str = lambda: "mxnet"
current_backend_str.__name__ = "current_backend_str"
