"""Collection of CUPY general functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Union, Sequence, Callable
import cupy as cp
from operator import mul
from functools import reduce
import multiprocessing as _multiprocessing
from numbers import Number

# local
import ivy
from ivy.functional.backends.numpy.device import _to_device


def array_equal(x0: cp.ndarray, x1: cp.ndarray, /) -> bool:
    return cp.array_equal(x0, x1)


def container_types():
    return []


def current_backend_str() -> str:
    return "cupy"


def get_item(x: cp.ndarray, query: cp.ndarray) -> cp.ndarray:
    return x.__getitem__(query)


def to_numpy(x: cp.ndarray, /, *, copy: bool = True) -> cp.ndarray:
    if copy:
        return x.copy()
    else:
        return x


def to_scalar(x: cp.ndarray, /) -> Number:
    return x.item()


def to_list(x: cp.ndarray, /) -> list:
    return x.tolist()


def gather(
    params: cp.ndarray,
    indices: cp.ndarray,
    /,
    *,
    axis: Optional[int] = -1,
    batch_dims: Optional[int] = 0,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    result = []
    if batch_dims == 0:
        result = cp.take(params, indices, axis)
    else:
        for b in range(batch_dims):
            if b == 0:
                zip_list = [(p, i) for p, i in zip(params, indices)]
            else:
                zip_list = [
                    (p, i) for z in [zip(p1, i1) for p1, i1 in zip_list] for p, i in z
                ]
        for z in zip_list:
            p, i = z
            r = cp.take(p, i, axis - batch_dims)
            result.append(r)
        result = cp.array(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return _to_device(result)


def gather_nd(
    params: cp.ndarray, indices: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    indices_shape = indices.shape
    params_shape = params.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = cp.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = cp.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = cp.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = cp.tile(
        cp.reshape(cp.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = cp.tile(
        cp.expand_dims(cp.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = cp.reshape(indices_for_flat, (-1,)).astype(cp.int32)
    flat_gather = cp.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = cp.reshape(flat_gather, new_shape)
    return _to_device(res)


def get_num_dims(x, /, *, as_array=False):
    return cp.asarray(len(cp.shape(x))) if as_array else len(x.shape)


def inplace_arrays_supported():
    return True


def inplace_decrement(
    x: Union[ivy.Array, cp.ndarray], val: Union[ivy.Array, cp.ndarray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, cp.ndarray], val: Union[ivy.Array, cp.ndarray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_update(
    x: Union[ivy.Array, cp.ndarray],
    val: Union[ivy.Array, cp.ndarray],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    if ivy.is_array(x) and ivy.is_array(val):
        (x_native, val_native), _ = ivy.args_to_native(x, val)

        # make both arrays contiguous if not already
        if not x_native.flags.c_contiguous:
            x_native = cp.ascontiguousarray(x_native)
        if not val_native.flags.c_contiguous:
            val_native = cp.ascontiguousarray(val_native)

        if val_native.shape == x_native.shape:
            if x_native.dtype != val_native.dtype:
                x_native = x_native.astype(val_native.dtype)
            cp.copyto(x_native, val_native)
        else:
            x_native = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x
    else:
        return val


def inplace_variables_supported():
    return True


def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, cp.ndarray):
        return True
    return False


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def scatter_flat(
    indices: cp.ndarray,
    updates: cp.ndarray,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        ivy.assertions.check_equal(len(target.shape), 1)
        ivy.assertions.check_equal(target.shape[0], size)
    if reduction == "sum":
        if not target_given:
            target = cp.zeros([size], dtype=updates.dtype)
        cp.add.at(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = cp.zeros([size], dtype=updates.dtype)
        target = cp.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == "min":
        if not target_given:
            target = cp.ones([size], dtype=updates.dtype) * 1e12
        cp.minimum.at(target, indices, updates)
        if not target_given:
            target = cp.asarray(
                cp.where(target == 1e12, 0.0, target), dtype=updates.dtype
            )
    elif reduction == "max":
        if not target_given:
            target = cp.ones([size], dtype=updates.dtype) * -1e12
        cp.maximum.at(target, indices, updates)
        if not target_given:
            target = cp.asarray(
                cp.where(target == -1e12, 0.0, target), dtype=updates.dtype
            )
    else:
        raise ivy.exceptions.IvyException(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target)


def scatter_nd(
    indices: cp.ndarray,
    updates: cp.ndarray,
    /,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and target_given:
        ivy.assertions.check_equal(ivy.Shape(target.shape), ivy.Shape(shape))
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
    if indices is not Ellipsis and (
        isinstance(indices, (tuple, list)) and not (Ellipsis in indices)
    ):
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = cp.array(indices)
        if len(indices.shape) < 2:
            indices = cp.expand_dims(indices, -1)
        expected_shape = (
            indices.shape[:-1] + out.shape[indices.shape[-1] :]
            if ivy.exists(out)
            else indices.shape[:-1] + tuple(shape[indices.shape[-1] :])
        )
        if sum(updates.shape) < sum(expected_shape):
            updates = ivy.broadcast_to(updates, expected_shape)._data
        elif sum(updates.shape) > sum(expected_shape):
            indices = ivy.broadcast_to(
                indices, updates.shape[:1] + (indices.shape[-1],)
            )._data
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if reduction == "sum":
        if not target_given:
            target = cp.zeros(shape, dtype=updates.dtype)
        cp.add.at(target, indices_tuple, updates)
    elif reduction == "replace":
        if not target_given:
            target = cp.zeros(shape, dtype=updates.dtype)
        target = cp.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == "min":
        if not target_given:
            target = cp.ones(shape) * 1e12
        cp.minimum.at(target, indices_tuple, updates)
        if not target_given:
            target = cp.where(target == 1e12, 0, target)
            target = cp.asarray(target, dtype=updates.dtype)
    elif reduction == "max":
        if not target_given:
            target = cp.ones(shape, dtype=updates.dtype) * -1e12
        cp.maximum.at(target, indices_tuple, updates)
        if not target_given:
            target = cp.where(target == -1e12, 0.0, target)
            target = cp.asarray(target, dtype=updates.dtype)
    else:
        raise ivy.exceptions.IvyException(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    if ivy.exists(out):
        return ivy.inplace_update(out, _to_device(target))
    return _to_device(target)


scatter_nd.support_native_out = True


def shape(x: cp.ndarray, /, *, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(cp.shape(x), dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    @ivy.to_native_arrays_and_back
    def _vmap(*args):

        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            ivy.assertions.check_equal(
                len(args),
                len(in_axes),
                message="""in_axes should have a length equivalent to the number
                of positional arguments to the function being vectorized or it
                should be an integer""",
            )

        # checking uniqueness of axis_size
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for arg, axis in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        if len(axis_size) > 1:
            raise ivy.exceptions.IvyException(
                """Inconsistent sizes. All mapped axes should have the same size"""
            )

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            ivy.assertions.check_any(
                [ivy.exists(ax) for ax in in_axes],
                message="At least one of the axes should be specified (not None)",
            )
        else:
            ivy.assertions.check_exists(
                in_axes, message="single value in_axes should not be None"
            )

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = cp.broadcast_to(
                    args[none_mapped_axis],
                    (tuple(axis_size) + args[none_mapped_axis].shape),
                )

        # set up the axis to be mapped to index zero.
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                if in_axes[i] is not None:
                    args[i] = cp.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = cp.moveaxis(args[0], in_axes, 0)

        # vectorisation. To be optimized.
        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = cp.stack(arr_results)

        if out_axes:
            res = cp.moveaxis(res, 0, out_axes)

        return res

    return _vmap
