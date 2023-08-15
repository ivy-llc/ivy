"""Collection of Numpy general functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Union, Sequence, Callable, Tuple
import numpy as np
from operator import mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing
from numbers import Number

# local
import ivy
from ivy.functional.backends.numpy.device import _to_device
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ...ivy.general import _broadcast_to


def array_equal(x0: np.ndarray, x1: np.ndarray, /) -> bool:
    return np.array_equal(x0, x1)


def container_types():
    return []


def current_backend_str() -> str:
    return "numpy"


@_scalar_output_to_0d_array
def get_item(
    x: np.ndarray,
    /,
    query: Union[np.ndarray, Tuple],
    *,
    copy: bool = None,
) -> np.ndarray:
    if copy:
        return x.__getitem__(query).copy()
    return x.__getitem__(query)


@_scalar_output_to_0d_array
def set_item(
    x: np.ndarray,
    query: Union[np.ndarray, Tuple],
    val: np.ndarray,
    /,
    *,
    copy: Optional[bool] = False,
) -> np.ndarray:
    if copy:
        x = np.copy(x)
    x.__setitem__(query, val)
    return x


def to_numpy(x: np.ndarray, /, *, copy: bool = True) -> np.ndarray:
    if copy:
        return x.copy()
    else:
        return x


def to_scalar(x: np.ndarray, /) -> Number:
    if isinstance(x, (float, int)):
        return x
    return x.item()


def to_list(x: np.ndarray, /) -> list:
    return x.tolist()


def gather(
    params: np.ndarray,
    indices: np.ndarray,
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    axis = axis % len(params.shape)
    batch_dims = batch_dims % len(params.shape)
    ivy.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    result = []
    if batch_dims == 0:
        result = np.take(params, indices, axis)
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
            r = np.take(p, i, axis - batch_dims)
            result.append(r)
        result = np.array(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return _to_device(result)


def gather_nd_helper(params, indices):
    indices_shape = indices.shape
    params_shape = params.shape
    if len(indices.shape) == 0:
        num_index_dims = 1
    else:
        num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        _reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = np.array(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = np.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = np.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = np.tile(
        np.reshape(np.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = np.tile(
        np.expand_dims(np.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = np.reshape(indices_for_flat, (-1,)).astype(np.int32)
    flat_gather = np.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    res = np.reshape(flat_gather, new_shape)
    return res


def gather_nd(
    params: np.ndarray,
    indices: np.ndarray,
    /,
    *,
    batch_dims: int = 0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ivy.utils.assertions.check_gather_nd_input_valid(params, indices, batch_dims)
    batch_dims = batch_dims % len(params.shape)
    result = []
    if batch_dims == 0:
        result = gather_nd_helper(params, indices)
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
            r = gather_nd_helper(p, np.asarray(i, indices.dtype))
            result.append(r)
        result = np.array(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return _to_device(result)


def get_num_dims(x, /, *, as_array=False):
    return np.asarray(len(np.shape(x))) if as_array else len(x.shape)


def inplace_arrays_supported():
    return True


def inplace_decrement(
    x: Union[ivy.Array, np.ndarray], val: Union[ivy.Array, np.ndarray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, np.ndarray], val: Union[ivy.Array, np.ndarray]
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_update(
    x: Union[ivy.Array, np.ndarray],
    val: Union[ivy.Array, np.ndarray],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> ivy.Array:
    ivy.utils.assertions.check_inplace_sizes_valid(x, val)
    if ivy.is_array(x) and ivy.is_array(val):
        if keep_input_dtype:
            val = ivy.astype(val, x.dtype)
        (x_native, val_native), _ = ivy.args_to_native(x, val)

        # make both arrays contiguous if not already
        if not x_native.flags.c_contiguous:
            x_native = np.ascontiguousarray(x_native)
        if not val_native.flags.c_contiguous:
            val_native = np.ascontiguousarray(val_native)

        if val_native.shape == x_native.shape:
            if x_native.dtype != val_native.dtype:
                x_native = x_native.astype(val_native.dtype)
            np.copyto(x_native, val_native)
        else:
            x_native = val_native
        if ivy.is_native_array(x):
            return x_native
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
    if isinstance(x, (np.ndarray, np.generic)):
        return True
    return False


def multiprocessing(context: Optional[str] = None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def scatter_flat(
    indices: np.ndarray,
    updates: np.ndarray,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        ivy.utils.assertions.check_equal(len(target.shape), 1, as_array=False)
        ivy.utils.assertions.check_equal(target.shape[0], size, as_array=False)
    if not target_given:
        reduction = "replace"
    if reduction == "sum":
        np.add.at(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = np.zeros([size], dtype=updates.dtype)
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices] = updates
    elif reduction == "min":
        np.minimum.at(target, indices, updates)
    elif reduction == "max":
        np.maximum.at(target, indices, updates)
    else:
        raise ivy.utils.exceptions.IvyException(
            "reduction is {}, but it must be one of "
            '"sum", "min", "max" or "replace"'.format(reduction)
        )
    if target_given:
        return ivy.inplace_update(out, target)
    return target


scatter_flat.support_native_out = True


def scatter_nd(
    indices: np.ndarray,
    updates: np.ndarray,
    /,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and target_given:
        ivy.utils.assertions.check_equal(
            ivy.Shape(target.shape), ivy.Shape(shape), as_array=False
        )
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    if not target_given:
        shape = list(shape) if ivy.exists(shape) else list(out.shape)
        target = np.zeros(shape, dtype=updates.dtype)
    updates = _broadcast_to(updates, target[indices_tuple].shape)
    if reduction == "sum":
        np.add.at(target, indices_tuple, updates)
    elif reduction == "replace":
        target = np.asarray(target).copy()
        target.setflags(write=1)
        target[indices_tuple] = updates
    elif reduction == "min":
        np.minimum.at(target, indices_tuple, updates)
    elif reduction == "max":
        np.maximum.at(target, indices_tuple, updates)
    else:
        raise ivy.utils.exceptions.IvyException(
            "reduction is {}, but it must be one of "
            '"sum", "min", "max" or "replace"'.format(reduction)
        )
    if ivy.exists(out):
        return ivy.inplace_update(out, _to_device(target))
    return _to_device(target)


scatter_nd.support_native_out = True


def shape(
    x: np.ndarray,
    /,
    *,
    as_array: bool = False,
) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(np.shape(x), dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    @ivy.output_to_native_arrays
    @ivy.inputs_to_native_arrays
    def _vmap(*args):
        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            ivy.utils.assertions.check_equal(
                len(args),
                len(in_axes),
                message="""in_axes should have a length equivalent to the number
                of positional arguments to the function being vectorized or it
                should be an integer""",
                as_array=False,
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
            raise ivy.utils.exceptions.IvyException(
                """Inconsistent sizes. All mapped axes should have the same size"""
            )

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            ivy.utils.assertions.check_any(
                [ivy.exists(ax) for ax in in_axes],
                message="At least one of the axes should be specified (not None)",
                as_array=False,
            )
        else:
            ivy.utils.assertions.check_exists(
                in_axes, message="single value in_axes should not be None"
            )

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = np.broadcast_to(
                    args[none_mapped_axis],
                    (tuple(axis_size) + args[none_mapped_axis].shape),
                )

        # set up the axis to be mapped to index zero.
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                if in_axes[i] is not None:
                    args[i] = np.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = np.moveaxis(args[0], in_axes, 0)

        # vectorisation. To be optimized.
        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = np.stack(arr_results)

        if out_axes:
            res = np.moveaxis(res, 0, out_axes)

        return res

    return _vmap


@with_unsupported_dtypes({"1.25.2 and below": ("bfloat16",)}, backend_version)
def isin(
    elements: np.ndarray,
    test_elements: np.ndarray,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> np.ndarray:
    return np.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True


def itemsize(x: np.ndarray) -> int:
    return x.itemsize
