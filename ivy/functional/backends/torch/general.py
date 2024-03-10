"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and
signature."""

# global
from functools import reduce as _reduce
import functools
from numbers import Number
from operator import mul
from typing import Callable, List, Optional, Sequence, Tuple, Union

try:
    import functorch
except ImportError:
    functorch = ()  # for torch 1.10.1
import numpy as np
import torch

# local
import ivy
from ivy.func_wrapper import _update_torch_views, with_unsupported_dtypes

from ...ivy.general import _broadcast_to
from . import backend_version, is_variable

torch_scatter = None


def _parse_index(indices, ndims):
    ind = []
    for so in indices:
        pre = []
        for s in so:
            if s == -1:
                break
            pre.append(s.item())
        post = []
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


@with_unsupported_dtypes({"2.2 and below": ("complex", "bfloat16")}, backend_version)
def array_equal(x0: torch.Tensor, x1: torch.Tensor, /) -> bool:
    x0, x1 = ivy.promote_types_of_inputs(x0, x1)
    return torch.equal(x0, x1)


def container_types():
    return []


def current_backend_str() -> str:
    return "torch"


def neg_step(query):
    return (
        not isinstance(query, (int, bool))
        and not ivy.is_array(query)
        and query is not None
        and query is not Ellipsis
        and (
            (isinstance(query, slice) and query.step is not None and query.step < 0)
            or (
                not isinstance(query, slice)
                and any(
                    isinstance(q, slice) and q.step is not None and q.step < 0
                    for q in query
                )
            )
        )
    )


def get_item(
    x: torch.Tensor,
    /,
    query: Union[torch.Tensor, Tuple],
    *,
    copy: Optional[bool] = None,
) -> torch.Tensor:
    if copy:
        x = x.clone()
    return x.__getitem__(query)


get_item.partial_mixed_handler = lambda x, query, **kwargs: not neg_step(query)


def set_item(
    x: torch.Tensor,
    query: Union[torch.Tensor, Tuple],
    val: torch.Tensor,
    /,
    *,
    copy: Optional[bool] = False,
) -> torch.Tensor:
    if hasattr(x, "dtype") and hasattr(val, "dtype") and x.dtype != val.dtype:
        val = val.to(x.dtype)
    if copy:
        x = x.clone()
    x.__setitem__(query, val)
    return x


set_item.partial_mixed_handler = (
    lambda x, query, val, **kwargs: not neg_step(query) and not x.requires_grad
)


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
            # we don't use inbuilt numpy() because it blocks for
            # bfloat16, which we are supporting here by importing
            # ml_dtypes
            # TODO: use torch's numpy() method once this feature is accepted
            # https://github.com/pytorch/pytorch/issues/109873
            if 0 in x.shape:
                # this is necessary because tolist converts all empty shapes to (0,)
                return np.empty(x.shape, dtype=ivy.as_ivy_dtype(x.dtype))
            return np.array(x.tolist(), dtype=ivy.as_ivy_dtype(x.dtype))
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
    axis %= len(params.shape)
    batch_dims %= len(params.shape)
    ivy.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    result = []

    def expand_p_i(params, indices, axis=axis, batch_dims=batch_dims):
        axis %= len(params.shape)
        abs(params.dim() - (indices.dim() - batch_dims))
        stack_dims1 = params.shape[:axis]
        stack_dims2 = params.shape[axis + 1 :]
        indices = indices.reshape(
            (
                torch.Size([1 for dim in stack_dims1])
                + torch.Size([-1])
                + torch.Size([1 for dim in stack_dims2])
            )
        )
        indices = indices.expand(
            (stack_dims1 + torch.Size([-1]) + stack_dims2)
        ).reshape((stack_dims1 + torch.Size([-1]) + stack_dims2))
        return indices, axis

    final_shape = (
        params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis + 1 :]
    )

    if batch_dims == 0:
        dim_diff = abs(params.dim() - (indices.dim() - batch_dims))
        if dim_diff != 0:
            indices_expanded, new_axis = expand_p_i(params, indices)

            result = torch.gather(
                params, new_axis, indices_expanded.long(), sparse_grad=False, out=out
            ).reshape(
                params.shape[:axis]
                + indices.shape[batch_dims:]
                + params.shape[axis + 1 :]
            )
            result = result.to(dtype=params.dtype)
            return result
        else:
            indices_expanded, new_axis = expand_p_i(params, indices)
            result = torch.gather(
                params, new_axis, indices_expanded.long(), sparse_grad=False, out=out
            ).reshape(final_shape)
            result = result.to(dtype=params.dtype)

    else:
        indices_ex = indices
        new_axis = axis
        params_slices = torch.unbind(params, axis=0) if params.shape[0] > 0 else params
        indices_slices = (
            torch.unbind(indices_ex, axis=0) if indices.shape[0] > 0 else indices_ex
        )
        for b in range(batch_dims):
            if b == 0:
                zip_list = [(p, i) for p, i in zip(params_slices, indices_slices)]
            else:
                zip_list = [
                    (p, i) for z in [zip(p1, i1) for p1, i1 in zip_list] for p, i in z
                ]
        for z in zip_list:
            p, i = z
            i_ex, new_axis = expand_p_i(p, i, axis=axis - batch_dims)
            r = torch.gather(p, (new_axis), i_ex.long(), sparse_grad=False, out=None)
            result.append(r)
        result = torch.stack(result)
        result = result.reshape(
            params.shape[:axis]
            + max(indices.shape[batch_dims:], torch.Size([1]))
            + params.shape[axis + 1 :]
        )
        result = result.to(dtype=params.dtype)
        if ivy.exists(out):
            return ivy.inplace_update(out, result)
    return result


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
    batch_dims %= len(params.shape)
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


def size(x: torch.Tensor, /) -> int:
    return functools.reduce(mul, x.shape) if len(x.shape) > 0 else 1


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
            x_native.copy_ = val_native
        else:
            x_native[()] = val_native
        x_native = x_native.to(val_native.device)
        if ivy.is_native_array(x):
            return x_native
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
        "2.2 and below": ("bfloat16",),
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
        ivy.utils.assertions.check_equal(len(target.shape), 1, as_array=False)
        ivy.utils.assertions.check_equal(target.shape[0], size, as_array=False)
    dtype = updates.dtype
    if reduction not in ["sum", "replace", "min", "max"]:
        raise ivy.utils.exceptions.IvyException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max" or'
            ' "replace"'
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
        except ImportError as e:
            raise ivy.utils.exceptions.IvyException(
                "Unable to import torch_scatter, verify this is correctly installed."
            ) from e
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
        "2.2 and below": (
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
    updates = torch.tensor(
        updates,
        dtype=(
            ivy.dtype(out, as_native=True)
            if ivy.exists(out)
            else ivy.default_dtype(item=updates, as_native=True)
        ),
    )

    expected_shape = (
        list(indices.shape[:-1]) + list(out.shape[indices.shape[-1] :])
        if ivy.exists(out)
        else list(indices.shape[:-1]) + list(shape[indices.shape[-1] :])
    )
    updates = _broadcast_to(updates, expected_shape)._data

    # implementation
    target_given = ivy.exists(out)
    if ivy.exists(shape) and target_given:
        ivy.utils.assertions.check_equal(
            ivy.Shape(out.shape), ivy.Shape(shape), as_array=False
        )
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
    dtype = updates.dtype
    indices_shape = indices.shape
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        _reduce(mul, shape[i + 1 :], 1) for i in range(len(shape) - 1)
    ] + [1]
    result_dim_sizes = torch.tensor(result_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_result_size = _reduce(mul, shape, 1)
    if reduction not in ["sum", "replace", "min", "max"]:
        raise ivy.utils.exceptions.IvyException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max" or'
            ' "replace"'
        )
    if target_given:
        flat_output = torch.reshape(out, (flat_result_size,)).detach()
    else:
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
        except ImportError as e:
            raise ivy.utils.exceptions.IvyException(
                "Unable to import torch_scatter, verify this is correctly installed."
            ) from e
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


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, backend_version)
def vmap_v_1p13p1_and_below(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    @ivy.output_to_native_arrays
    @ivy.inputs_to_native_arrays
    def _vmap(*args):
        def new_fun(*args):
            return ivy.to_native(func(*args))

        new_func = functorch.vmap(new_fun, in_axes, out_axes)
        return new_func(*args)

    return _vmap


@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, backend_version)
def vmap_v_2p0p0_and_above(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    @ivy.output_to_native_arrays
    @ivy.inputs_to_native_arrays
    def _vmap(*args):
        def new_fun(*args):
            return ivy.to_native(func(*args))

        new_func = torch.vmap(new_fun, in_axes, out_axes)
        return new_func(*args)

    return _vmap


@with_unsupported_dtypes(
    {"2.2 and below": ("bfloat16", "float16", "complex", "bool")}, backend_version
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
