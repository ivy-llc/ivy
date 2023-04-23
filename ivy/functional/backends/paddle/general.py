"""Collection of Paddle general functions, wrapped to fit Ivy syntax and signature."""
# global
from numbers import Number
from typing import Optional, Union, Sequence, Callable, List, Tuple
import paddle
import numpy as np

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version
import multiprocessing as _multiprocessing
from ivy.functional.ivy.general import _parse_ellipsis, _parse_index


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, paddle.Tensor):
        if exclusive and not x.stop_gradient:
            return False
        return True
    return False


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def array_equal(x0: paddle.Tensor, x1: paddle.Tensor, /) -> bool:
    return bool(ivy.all(ivy.equal(x0, x1)))


def container_types():
    return []


def current_backend_str() -> str:
    return "paddle"


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def get_item(
    x: paddle.Tensor, query: Union[paddle.Tensor, Tuple], *, copy: bool = None
) -> paddle.Tensor:
    # regular queries x[idx_1,idx_2,...,idx_i]
    if not isinstance(query, paddle.Tensor):
        if x.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
            return x.cast("float32").__getitem__(query).cast(x.dtype)
        return x.__getitem__(query)

    # masked queries x[bool_1,bool_2,...,bool_i]
    if query.dtype == paddle.bool:
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
                return paddle.complex(
                    paddle.masked_select(x.real(), query),
                    paddle.masked_select(x.imag(), query),
                )
            return paddle.masked_select(x.cast("float32"), query).cast(x.dtype)
        return paddle.masked_select(x, query)

    query = query.cast("int64")
    # array queries idx = Tensor(idx_1,idx_2,...,idx_i), x[idx]
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
            return paddle.complex(
                x.real().__getitem__(query), x.imag().__getitem__(query)
            )
        return x.cast("float32").__getitem__(query).cast(x.dtype)
    return x.__getitem__(query)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def to_numpy(
    x: Union[paddle.Tensor, List[paddle.Tensor]], /, *, copy: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(x, (float, int, bool)):
        return x
    elif isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    elif paddle.is_tensor(x):
        if copy:
            return np.array(x)
        else:
            return np.asarray(x)
    elif isinstance(x, list):
        return [ivy.to_numpy(u) for u in x]
    raise ivy.utils.exceptions.IvyException("Expected a Paddle Tensor.")


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def to_scalar(x: paddle.Tensor, /) -> Number:
    if isinstance(x, (Number, complex)):
        return x
    return x.item()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def to_list(x: paddle.Tensor, /) -> list:
    return x.tolist()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def gather(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = -1,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    def _gather(params1):
        with ivy.ArrayMode(False):
            if batch_dims == 0:
                result = paddle.gather(
                    params1, ivy.reshape(indices, shape=[-1]), axis=axis
                )
            # inputs are unstacked batch_dims times
            # because paddle.gather does not support batch_dims
            else:
                params1_list = ivy.unstack(params1, axis=0)
                indices_list = ivy.unstack(indices, axis=0)
                for b in range(1, batch_dims):
                    params1_list = [
                        p2 for p1 in params1_list for p2 in ivy.unstack(p1, axis=0)
                    ]
                    indices_list = [
                        i2 for i1 in indices_list for i2 in ivy.unstack(i1, axis=0)
                    ]
                result = []
                for p, i in zip(params1_list, indices_list):
                    result.append(
                        paddle.gather(
                            p, ivy.reshape(i, shape=[-1]), axis=axis - batch_dims
                        )
                    )
                result = ivy.concat(result, axis=0)
            new_shape = (
                params1.shape[:axis]
                + indices.shape[batch_dims:]
                + params1.shape[axis + 1 :]
            )
            return ivy.reshape(result, shape=new_shape)

    axis = axis % params.ndim
    batch_dims = batch_dims % params.ndim
    ivy.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    if params.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(params):
            return paddle.complex(_gather(params.real()), _gather(params.imag()))
        return _gather(params.cast("float32")).cast(params.dtype)
    return _gather(params)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def gather_nd(
    params: paddle.Tensor,
    indices: paddle.Tensor,
    /,
    *,
    batch_dims: Optional[int] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def get_num_dims(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[paddle.Tensor, int]:
    return paddle.to_tensor(x.ndim) if as_array else x.ndim


def inplace_arrays_supported():
    # there are some operations that support inplace updates
    # but it's not supported in all functions
    return False


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_decrement(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_increment(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_update(
    x: Union[ivy.Array, paddle.Tensor],
    val: Union[ivy.Array, paddle.Tensor],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> ivy.Array:
    if ivy.is_array(x) and ivy.is_array(val):
        (x_native, val_native), _ = ivy.args_to_native(x, val)

        if val_native.shape == x_native.shape:
            if x_native.dtype != val_native.dtype:
                x_native = x_native.astype(val_native.dtype)
            paddle.assign(val_native, x_native)
        else:
            x_native = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
        else:
            x = ivy.Array(x_native)
        return x
    else:
        return val


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def inplace_variables_supported():
    return False


def multiprocessing(context=None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def scatter_flat(
    indices: paddle.Tensor,
    updates: paddle.Tensor,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def _scatter_nd_replace(data, indices, updates, reduce):
    """
    This is an implementation for scatter_nd using put_along_axis
    since paddle.scatter_nd only supports "sum" reduction mode
    """
    if data.dtype != updates.dtype:
        data = data.cast(updates.dtype)

    target_idx = paddle.to_tensor([])
    idx_range = paddle.arange(paddle.prod(paddle.to_tensor(data.shape))).reshape(
        data.shape
    )

    for i in indices:
        target_idx = paddle.concat(
            [target_idx, ivy.get_item(idx_range, tuple(i)).flatten()], -1
        )

    target_idx = target_idx.reshape([1] * (data.ndim - 1) + target_idx.shape[-1:])

    if data.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(data):
            result_real = paddle.put_along_axis(
                data.real(), target_idx, updates.real().flatten(), -1, reduce=reduce
            )
            result_imag = paddle.put_along_axis(
                data.imag(), target_idx, updates.imag().flatten(), -1, reduce=reduce
            )
            return paddle.complex(result_real, result_imag)
        return paddle.put_along_axis(
            data.cast("float32"),
            target_idx,
            updates.cast("float32").flatten(),
            -1,
            reduce=reduce,
        ).cast(data.dtype)
    return paddle.put_along_axis(data, target_idx, updates.flatten(), -1, reduce=reduce)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def scatter_nd(
    indices: paddle.Tensor,
    updates: paddle.Tensor,
    /,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        if ivy.exists(out) and not isinstance(updates, (Number, list, tuple)):
            out = (
                out.cast(updates.dtype)
                if ivy.dtype_bits(updates.dtype) > ivy.dtype_bits(out.dtype)
                else out
            )
        # handle numeric updates
        updates = ivy.array(updates)
        updates = updates.cast(
            ivy.dtype(out, as_native=True)
            if ivy.exists(out)
            else ivy.default_dtype(item=updates),
        )
        contains_slices = (
            any(isinstance(idx, slice) for idx in indices)
            if isinstance(indices, (tuple, list))
            else isinstance(indices, slice)
        )

        # hanle non-tensor indices
        if len(indices) == 0:
            return updates

        elif (
            indices is Ellipsis
            or (isinstance(indices, tuple) and indices == (Ellipsis,))
            or (isinstance(indices, slice) and indices == slice(None, None, None))
        ):
            if updates.shape == () and ivy.exists(out) and out.shape == ():
                return updates
            shape = out.shape if ivy.exists(out) else updates.shape
            indices = ivy.stack(
                [
                    ivy.reshape(value, (-1,))
                    for value in ivy.meshgrid(*[ivy.arange(shape[0])], indexing="ij")
                ],
                axis=-1,
            )
        elif isinstance(indices, (tuple, list)) and Ellipsis in indices:
            shape = (
                shape
                if ivy.exists(shape)
                else out.shape
                if ivy.exists(out)
                else updates.shape
            )
            indices = _parse_ellipsis(indices, len(shape))
            indices = ivy.stack(
                [
                    ivy.reshape(value, (-1,))
                    for value in ivy.meshgrid(
                        *[
                            ivy.arange(s)
                            if idx == slice(None, None, None)
                            else ivy.arange(
                                ivy.default(idx.start, 0),
                                ivy.default(idx.stop, shape[0]),
                                ivy.default(idx.step, 1),
                            )
                            if isinstance(idx, slice)
                            and (idx != slice(None, None, None))
                            else ivy.array([idx % s])
                            for s, idx in zip(shape, indices)
                        ],
                        indexing="ij",
                    )
                ],
                axis=-1,
            )
        elif contains_slices:
            shape = (
                shape
                if ivy.exists(shape)
                else out.shape
                if ivy.exists(out)
                else updates.shape
            )
            if isinstance(indices, (tuple, list)):
                indices = (
                    _parse_index(indices, len(shape)) if -1 in indices else indices
                )
                indices = ivy.stack(
                    [
                        ivy.reshape(value, (-1,))
                        for value in ivy.meshgrid(
                            *[
                                ivy.arange(s)
                                if idx == slice(None, None, None)
                                else ivy.arange(
                                    ivy.default(idx.start, 0),
                                    ivy.default(idx.stop, shape[0]),
                                    ivy.default(idx.step, 1),
                                )
                                if isinstance(idx, slice)
                                and (idx != slice(None, None, None))
                                else ivy.array([idx % s])
                                for s, idx in zip(shape, indices)
                            ],
                            indexing="ij",
                        )
                    ],
                    axis=-1,
                )
            else:
                indices = ivy.stack(
                    [
                        ivy.reshape(value, (-1,))
                        for value in ivy.meshgrid(
                            *[
                                ivy.arange(
                                    ivy.default(indices.start, 0),
                                    ivy.default(indices.stop, shape[0]),
                                    ivy.default(indices.step, 1),
                                )
                            ],
                            indexing="ij",
                        )
                    ],
                    axis=-1,
                )
        else:
            indices = [[indices]] if isinstance(indices, Number) else indices
            indices = ivy.array(indices).data
            if len(indices.shape) < 2:
                indices = ivy.expand_dims(indices, 0)
            if ivy.any(indices < 0):
                shape = list(shape) if ivy.exists(shape) else list(out.shape)
                indices = _parse_index(indices, shape)
                indices = [
                    ivy.stack(
                        [
                            ivy.reshape(value, (-1,))
                            for value in ivy.meshgrid(
                                *[
                                    ivy.arange(s)
                                    if idx == slice(None, None, None)
                                    else ivy.arange(
                                        ivy.default(idx.start, 0),
                                        ivy.ivy.default(idx.stop, shape[0]),
                                        ivy.default(idx.step, 1),
                                    )
                                    if isinstance(idx, slice)
                                    and idx != slice(None, None, None)
                                    else ivy.array([idx % s])
                                    for s, idx in zip(shape, index)
                                ],
                                indexing="xy",
                            )
                        ],
                        axis=-1,
                    )
                    for index in indices
                ]
                indices = ivy.concat(indices, axis=0)
        # broadcast updates to correct shape
        shape = list(shape) if shape is not None else None
        expected_shape = (
            indices.shape[:-1] + list(out.shape[indices.shape[-1] :])
            if ivy.exists(out)
            else indices.shape[:-1] + shape[indices.shape[-1] :]
        )
        if sum(updates.shape) < sum(expected_shape):
            updates = ivy.broadcast_to(updates, expected_shape)
        elif sum(updates.shape) > sum(expected_shape):
            indices = ivy.broadcast_to(indices, updates.shape[:1] + indices.shape[-1])
        elif updates.shape != expected_shape:
            updates = ivy.broadcast_to(updates, expected_shape)
        # implementation
        target = out
        target_given = ivy.exists(target)
        if ivy.exists(shape) and ivy.exists(target):
            ivy.utils.assertions.check_equal(ivy.Shape(target.shape), ivy.Shape(shape))
        shape = list(shape) if ivy.exists(shape) else out.shape
        if not target_given:
            target = ivy.zeros(shape=shape, dtype=updates.dtype)
            ret = _scatter_nd_replace(target, indices, updates, reduce="assign")

        if reduction == "sum":
            ret = _scatter_nd_replace(target, indices, updates, reduce="add")

        elif reduction == "min":
            new_updates = paddle.to_tensor([], dtype=target.dtype)
            for i in indices:
                new_updates = ivy.concat(
                    [new_updates, ivy.get_item(target, tuple(i)).reshape([-1])], axis=-1
                )
            new_updates = ivy.minimum(new_updates, updates.reshape([-1]))
            ret = _scatter_nd_replace(target, indices, new_updates, reduce="assign")

        elif reduction == "max":
            new_updates = paddle.to_tensor([], dtype=target.dtype)
            for i in indices:
                new_updates = ivy.concat(
                    [new_updates, ivy.get_item(target, tuple(i)).reshape([-1])], axis=-1
                )
            new_updates = ivy.maximum(new_updates, updates.reshape([-1]))
            ret = _scatter_nd_replace(target, indices, new_updates, reduce="assign")
        elif reduction == "replace":
            ret = _scatter_nd_replace(target, indices, updates, reduce="assign")
        else:
            raise ivy.utils.exceptions.IvyException(
                'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                    reduction
                )
            )
        if ivy.exists(out):
            return paddle.assign(ret, out.data)
        else:
            return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def shape(
    x: paddle.Tensor, /, *, as_array: bool = False
) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(x.shape, dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: Optional[int] = 0,
) -> Callable:
    raise IvyNotImplementedException()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def isin(
    elements: paddle.Tensor,
    test_elements: paddle.Tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> paddle.Tensor:
    input_shape = elements.shape
    if elements.ndim == 0:
        elements = ivy.reshape(elements, [1])
    if test_elements.ndim == 0:
        test_elements = ivy.reshape(test_elements, [1])
    if not assume_unique:
        test_elements = ivy.unique_values(test_elements)

    elements = elements.reshape([-1])
    test_elements = test_elements.reshape([-1])

    output = ivy.any(
        ivy.equal(ivy.expand_dims(elements, axis=-1), test_elements), axis=-1
    )
    return ivy.reshape(output, input_shape) ^ invert


def itemsize(x: paddle.Tensor) -> int:
    return x.element_size()
