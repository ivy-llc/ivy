"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature."""
# global
from functools import reduce
import math
from numbers import Number
from operator import mul
from typing import Optional, Union, Sequence, Callable, List, Tuple

try:
    import functorch
except ImportError:
    functorch = ()  # for torch 1.10.1
import numpy as np
import torch

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, _update_torch_views
from ivy.functional.ivy.general import _parse_ellipsis
from . import backend_version, is_variable

torch_scatter = None


def _parse_index(indices, ndims):
    ind = list()
    for so in indices:
        pre = list()
        for s in so:
            if s == -1:
                break
            pre.append(s.item())
        post = list()
        for s in reversed(so):
            if s == -1:
                break
            post.append(s.item())
        ind.append(
            tuple(
                pre
                + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
                + list(reversed(post))
            )
        )
    return ind


def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, torch.Tensor):
        if exclusive and x.requires_grad:
            return False
        return True
    return False


@with_unsupported_dtypes({"1.11.0 and below": ("complex", "bfloat16")}, backend_version)
def array_equal(x0: torch.Tensor, x1: torch.Tensor, /) -> bool:
    x0, x1 = ivy.promote_types_of_inputs(x0, x1)
    return torch.equal(x0, x1)


def container_types():
    return []


def current_backend_str() -> str:
    return "torch"


def get_item(
    x: torch.Tensor,
    /,
    query: torch.Tensor,
    *,
    copy: bool = None,
) -> torch.Tensor:
    if copy:
        x = torch.clone(x)
    if ivy.is_array(query) and ivy.dtype(query, as_native=True) is not torch.bool:
        return x.__getitem__(query.to(torch.int64))
    if isinstance(query, slice) and query.step is not None and query.step < 0:
        start = query.start if query.start is not None else x.shape[0] - 1
        stop = query.stop if query.stop is not None else -1
        step = query.step if query.step is not None else -1
        return gather(x, torch.arange(start, stop, step, device=x.device), axis=0)
    try:
        return x.__getitem__(query)
    except ValueError:
        new_query = []
        shape_idx = 0
        for q in query:
            if isinstance(q, slice) and q.step is not None and q.step < 0:
                start = q.start if q.start is not None else x.shape[shape_idx] - 1
                stop = q.stop if q.stop is not None else -1
                step = q.step
                new_query.append(torch.arange(start, stop, step, device=x.device))
            else:
                new_query.append(q)
            shape_idx += 1
        return x.__getitem__(new_query)


def to_numpy(
    x: Union[torch.Tensor, List[torch.Tensor]], /, *, copy: bool = True
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(x, (float, int, bool)):
        return x
    elif isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    elif torch.is_tensor(x):
        x = x.resolve_neg().resolve_conj()
        if copy:
            if x.dtype is torch.bfloat16:
                default_dtype = ivy.default_float_dtype(as_native=True)
                if default_dtype is torch.bfloat16:
                    x = x.to(torch.float32)
                else:
                    x = x.to(default_dtype)
                return x.detach().cpu().numpy().astype("bfloat16")
            return x.detach().cpu().numpy()
        else:
            raise ivy.utils.exceptions.IvyException(
                "Overwriting the same address is not supported for torch."
            )
    elif isinstance(x, list):
        return [ivy.to_numpy(u) for u in x]
    raise ivy.utils.exceptions.IvyException("Expected a pytorch tensor.")


def to_scalar(x: torch.Tensor, /) -> Number:
    if isinstance(x, (float, int)):
        return x
    return x.item()


def to_list(x: torch.Tensor, /) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif torch.is_tensor(x):
        if x.dtype is torch.bfloat16:
            default_dtype = ivy.default_float_dtype(as_native=True)
            if default_dtype is torch.bfloat16:
                x = x.to(torch.float32)
            else:
                x = x.to(default_dtype)
            return x.detach().cpu().numpy().astype("bfloat16").tolist()
        else:
            return x.detach().cpu().numpy().tolist()
    raise ivy.utils.exceptions.IvyException("Expected a pytorch tensor.")


def gather(
    params: torch.Tensor,
    indices: torch.Tensor,
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    axis = axis % len(params.shape)
    batch_dims = batch_dims % len(params.shape)
    ivy.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    result = []
    if batch_dims == 0:
        result = params[
            (slice(None),) * (axis % params.ndim) + (indices.type(torch.int64),)
        ]
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
            r = p[
                (slice(None),) * ((axis - batch_dims) % p.ndim) + (i.type(torch.int64),)
            ]
            result.append(r)
        result = torch.stack(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return result


def gather_nd_helper(params, indices):
    indices_shape = indices.shape
    params_shape = params.shape
    if len(indices.shape) == 0:
        num_index_dims = 1
    else:
        num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = torch.tensor(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = torch.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = torch.reshape(
        torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)
    ).repeat(*[1, implicit_indices_factor])
    implicit_indices = torch.unsqueeze(torch.arange(implicit_indices_factor), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1]
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = torch.reshape(indices_for_flat, (-1,)).type(torch.long)
    flat_gather = torch.gather(flat_params, 0, flat_indices_for_flat)
    res = torch.reshape(
        flat_gather, list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    )
    return res


def gather_nd(
    params: torch.Tensor,
    indices: torch.Tensor,
    /,
    *,
    batch_dims: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
            r = gather_nd_helper(p, i)
            result.append(r)
        result = torch.stack(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return result


def get_num_dims(
    x: torch.Tensor, /, *, as_array: bool = False
) -> Union[torch.Tensor, int]:
    return torch.tensor(len(x.shape)) if as_array else len(x.shape)


def inplace_arrays_supported():
    return True


def inplace_decrement(
    x: Union[ivy.Array, torch.Tensor],
    val: Union[ivy.Array, torch.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native.data -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(
    x: Union[ivy.Array, torch.Tensor],
    val: Union[ivy.Array, torch.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native.data += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_update(
    x: Union[ivy.Array, torch.Tensor],
    val: Union[ivy.Array, torch.Tensor],
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
        if is_variable(x_native):
            x_native.data = val_native
        else:
            x_native[()] = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native
            _update_torch_views(x)
        else:
            x = ivy.to_ivy(x_native)
        if ensure_in_backend:
            x._data = val_native
        return x
    else:
        return val


def inplace_variables_supported():
    return True


def multiprocessing(context: Optional[str] = None):
    import torch.multiprocessing

    if context is None:
        return torch.multiprocessing
    return torch.multiprocessing.get_context(context)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("bfloat16",),
    },
    backend_version,
)
def scatter_flat(
    indices: torch.Tensor,
    updates: torch.Tensor,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        ivy.utils.assertions.check_equal(len(target.shape), 1)
        ivy.utils.assertions.check_equal(target.shape[0], size)
    dtype = updates.dtype
    if reduction not in ["sum", "replace", "min", "max"]:
        raise ivy.utils.exceptions.IvyException(
            "reduction is {}, but it must be one of "
            '"sum", "min", "max" or "replace"'.format(reduction)
        )
    if target_given:
        output = out
    else:
        reduction = "replace"
        output = torch.zeros([size], dtype=dtype)
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except ImportError:
            raise ivy.utils.exceptions.IvyException(
                "Unable to import torch_scatter, verify this is correctly installed."
            )
    if reduction == "replace":
        output[indices.type(torch.int64)] = updates
        res = output
    else:
        res = torch_scatter.scatter(
            updates, indices.type(torch.int64), out=output, reduce=reduction
        )
    return res


scatter_flat.support_native_out = True


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def scatter_nd(
    indices: torch.Tensor,
    updates: torch.Tensor,
    /,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # handle numeric updates
    updates = torch.tensor(
        [updates] if isinstance(updates, (float, int, bool)) else updates,
        dtype=(
            ivy.dtype(out, as_native=True)
            if ivy.exists(out)
            else ivy.default_dtype(item=updates, as_native=True)
        ),
    )

    # handle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(out) and out.shape == ():
            return updates
        shape = out.shape if ivy.exists(out) else updates.shape
        indices = torch.stack(
            [
                torch.reshape(value, (-1,))
                for value in torch.meshgrid(*[torch.range(0, shape[0] - 1)])
            ],
            dim=-1,
        )
    elif isinstance(indices, (tuple, list)) and Ellipsis in indices:
        shape = (
            shape
            if ivy.exists(shape)
            else out.shape if ivy.exists(out) else updates.shape
        )
        indices = _parse_ellipsis(indices, len(shape))
        indices = torch.stack(
            [
                torch.reshape(value, (-1,))
                for value in torch.meshgrid(
                    *[
                        (
                            torch.range(0, s - 1)
                            if idx == slice(None, None, None)
                            else torch.Tensor([idx % s])
                        )
                        for s, idx in zip(shape, indices)
                    ],
                    indexing="ij",
                )
            ],
            dim=-1,
        )
    else:
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = (
            torch.tensor(indices) if isinstance(indices, (tuple, list)) else indices
        )
        if len(indices.shape) < 2:
            indices = torch.unsqueeze(indices, 0)
        if torch.any(indices == -1):
            shape = (
                shape
                if ivy.exists(shape)
                else out.shape if ivy.exists(out) else updates.shape
            )
            indices = _parse_index(indices, len(shape))
            indices = [
                torch.stack(
                    [
                        torch.reshape(value, (-1,))
                        for value in torch.meshgrid(
                            *[
                                (
                                    torch.range(0, s - 1)
                                    if idx == slice(None, None, None)
                                    else torch.tensor([idx % s])
                                )
                                for s, idx in zip(shape, index)
                            ],
                            indexing="xy",
                        )
                    ],
                    dim=-1,
                )
                for index in indices
            ]
            indices = torch.concat(indices, axis=0)

    # broadcast updates to indices
    expected_shape = (
        indices.shape[:-1] + out.shape[indices.shape[-1] :]
        if ivy.exists(out)
        else indices.shape[:-1] + tuple(shape[indices.shape[-1] :])
    )
    if sum(updates.shape) < sum(expected_shape):
        updates = ivy.broadcast_to(updates, expected_shape)._data
    elif sum(updates.shape) > sum(expected_shape):
        indices = ivy.broadcast_to(indices, updates.shape[:1] + indices.shape[-1])._data

    # implementation
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        ivy.utils.assertions.check_equal(ivy.Shape(target.shape), ivy.Shape(shape))
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
    dtype = updates.dtype
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        reduce(mul, shape[i + 1 :], 1) for i in range(len(shape) - 1)
    ] + [1]
    result_dim_sizes = torch.tensor(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_result_size = reduce(mul, shape, 1)
    if reduction not in ["sum", "replace", "min", "max"]:
        raise ivy.utils.exceptions.IvyException(
            "reduction is {}, but it must be one of "
            '"sum", "min", "max" or "replace"'.format(reduction)
        )
    if target_given:
        flat_output = torch.reshape(out._data, (flat_result_size,))
    else:
        reduction = "replace"
        flat_output = torch.zeros(flat_result_size, dtype=dtype)
    flat_updates = torch.reshape(updates, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = torch.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = torch.reshape(
        torch.sum(indices * indices_scales, -1, keepdim=True), (-1, 1)
    ).repeat(*[1, implicit_indices_factor])
    implicit_indices = torch.unsqueeze(torch.arange(implicit_indices_factor), 0).repeat(
        *[indices_for_flat_tiled.shape[0], 1]
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = torch.reshape(indices_for_flat, (-1,)).type(torch.long)
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except ImportError:
            raise ivy.utils.exceptions.IvyException(
                "Unable to import torch_scatter, verify this is correctly installed."
            )
    if reduction == "replace":
        flat_output[flat_indices_for_flat] = flat_updates
        flat_scatter = flat_output
    else:
        flat_scatter = torch_scatter.scatter(
            flat_updates,
            flat_indices_for_flat,
            out=flat_output.clone(),
            reduce=reduction,
        )
    res = torch.reshape(flat_scatter, list(shape))
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res


scatter_nd.support_native_out = True


def shape(
    x: torch.Tensor,
    /,
    *,
    as_array: bool = False,
) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(x.shape, dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, backend_version)
def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    @ivy.output_to_native_arrays
    @ivy.inputs_to_native_arrays
    def _vmap(*args):
        new_fun = lambda *args: ivy.to_native(func(*args))
        new_func = functorch.vmap(new_fun, in_axes, out_axes)
        return new_func(*args)

    return _vmap


@with_unsupported_dtypes(
    {"1.11.0 and below": ("bfloat16", "float16", "complex", "bool")}, backend_version
)
def isin(
    elements: torch.tensor,
    test_elements: torch.tensor,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> torch.tensor:
    return torch.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True


def itemsize(x: torch.tensor) -> int:
    return x.element_size()


def strides(x: torch.tensor) -> Tuple[int]:
    return tuple(
        [int(stride * math.ceil(ivy.dtype_bits(x.dtype) / 8)) for stride in x.stride()]
    )
