"""Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature."""

# global
import ivy
import numpy as np
import torch
from operator import mul
from functools import reduce
from typing import List, Optional, Union, Sequence
from numbers import Number


torch_scatter = None


def is_native_array(x, exclusive=False):
    if isinstance(x, torch.Tensor):
        if exclusive and x.requires_grad:
            return False
        return True
    return False


def copy_array(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.clone()


def array_equal(x0: torch.Tensor, x1: torch.Tensor) -> bool:
    dtype = torch.promote_types(x0.dtype, x1.dtype)
    x0 = x0.type(dtype=dtype)
    x1 = x1.type(dtype=dtype)
    return torch.equal(x0, x1)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray) or isinstance(x, (float, int, bool)):
        return x
    elif torch.is_tensor(x):
        if x.dtype is torch.bfloat16:
            x = x.to(torch.float16)
        return x.detach().cpu().numpy()
    raise ValueError("Expected a pytorch tensor.")


def to_scalar(x: torch.Tensor) -> Number:
    if isinstance(x, (float, int)):
        return x
    return x.item()


def to_list(x: torch.Tensor) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif torch.is_tensor(x):
        return x.detach().cpu().tolist()
    raise ValueError("Expected a pytorch tensor.")


def floormod(
    x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    ret = x % y
    return ret


def unstack(x: torch.Tensor, axis: int, keepdims: bool = False) -> List[torch.Tensor]:
    if x.shape == ():
        return [x]
    ret = list(torch.unbind(x, axis))
    if keepdims:
        return [r.unsqueeze(axis) for r in ret]
    return ret


def container_types():
    return []


def inplace_update(
    x: Union[ivy.Array, torch.Tensor],
    val: Union[ivy.Array, torch.Tensor],
    ensure_in_backend: bool = False,
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native.data = val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.to_ivy(x_native)
    if ensure_in_backend:
        x._data = val_native
    return x


def inplace_arrays_supported():
    return True


inplace_variables_supported = lambda: True


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


def cumsum(
    x: torch.Tensor, axis: int = 0, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.cumsum(x, axis, out=out)


cumsum.support_native_out = True


def cumprod(
    x: torch.Tensor,
    axis: int = 0,
    exclusive: Optional[bool] = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if exclusive:
        x = torch.transpose(x, axis, -1)
        x = torch.cat((torch.ones_like(x[..., -1:]), x[..., :-1]), -1, out=out)
        res = torch.cumprod(x, -1, out=out)
        return torch.transpose(res, axis, -1)
    return torch.cumprod(x, axis, out=out)


cumprod.support_native_out = True


# noinspection PyShadowingNames
def scatter_flat(
    indices: torch.Tensor,
    updates: torch.Tensor,
    size: Optional[int] = None,
    reduction: str = "sum",
    *,
    out: Optional[torch.Tensor] = None,
):
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(size) and ivy.exists(target):
        assert len(target.shape) == 1 and target.shape[0] == size
    dtype = updates.dtype
    if reduction in ["sum", "replace"]:
        initial_val = torch.tensor(0).type(dtype)
    elif reduction == "min":
        initial_val = torch.tensor(1e12).type(dtype)
    elif reduction == "max":
        initial_val = torch.tensor(-1e12).type(dtype)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    if target_given:
        output = out
    else:
        output = torch.ones([size], dtype=dtype) * initial_val
    global torch_scatter
    if torch_scatter is None:
        try:
            import torch_scatter as torch_scatter
        except ImportError:
            raise Exception(
                "Unable to import torch_scatter, verify this is correctly installed."
            )
    if reduction == "replace":
        output[indices.type(torch.int64)] = updates
        res = output
    else:
        res = torch_scatter.scatter(
            updates, indices.type(torch.int64), out=output, reduce=reduction
        )
    if not target_given:
        return torch.where(
            res == initial_val,
            torch.zeros([size], dtype=updates.dtype),
            res,
        )
    return res


def _parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    return tuple(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )


# noinspection PyShadowingNames
def scatter_nd(
    indices: torch.Tensor,
    updates: torch.Tensor,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    reduction: str = "sum",
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # handle numeric updates
    updates = torch.tensor(
        [updates] if isinstance(updates, (float, int, bool)) else updates,
        dtype=ivy.dtype(out, as_native=True)
        if ivy.exists(out)
        else ivy.default_dtype(item=updates, as_native=True),
    )

    # hanle non-tensor indices
    if indices == ():
        return updates
    elif indices is Ellipsis or (isinstance(indices, tuple) and indices == (Ellipsis,)):
        if updates.shape == () and ivy.exists(out) and out.shape == ():
            return updates
        shape = out.shape if ivy.exists(out) else updates.shape
        indices =  torch.stack([torch.reshape(value, (-1,)) for value in torch.meshgrid(
                    *[
                        torch.range(0, shape[0]-1)  
                    ]
        )], axis=-1)
    elif isinstance(indices, (tuple, list)) and Ellipsis in indices:
        shape = out.shape if ivy.exists(out) else updates.shape
        indices = _parse_ellipsis(indices, len(shape))
        indices =   torch.stack([torch.reshape(value, (-1,)) for value in torch.meshgrid(
                    *[
                        torch.range(0, s-1) if idx == slice(None, None, None) else torch.Tensor([idx % s])
                        for s, idx in zip(shape, indices)
                    ], indexing ='ij'
        )], axis=-1)        
    else:
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = torch.Tensor(indices) if isinstance(indices, (tuple, list)) else indices
        if len(indices.shape) < 2:
                indices = torch.unsqueeze(indices, -1)
        
        if len(updates.shape) < 2:
            updates = torch.unsqueeze(updates, 0)
    
    # broadcast updates to indices
    if  updates.shape == ():
        updates = torch.broadcast_to(updates, indices.shape[:1]) 

    # implementation
    target = out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        assert ivy.Shape(target.shape) == ivy.Shape(shape)
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
    if reduction in ["sum", "replace"]:
        initial_val = torch.tensor(0).type(dtype)
    elif reduction == "min":
        initial_val = torch.tensor(1e12).type(dtype)
    elif reduction == "max":
        initial_val = torch.tensor(-1e12).type(dtype)
    else:
        raise Exception(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    if target_given:
        flat_output = torch.reshape(out._data, (flat_result_size,))
    else:
        flat_output = torch.ones(flat_result_size, dtype=dtype) * initial_val
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
            raise Exception(
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
    if not target_given:
        # noinspection PyTypeChecker
        flat_scatter = torch.where(
            flat_scatter == initial_val,
            torch.zeros(flat_result_size, dtype=updates.dtype),
            flat_scatter,
        )
    res = torch.reshape(flat_scatter, list(shape))
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res

scatter_nd.support_native_out = True
scatter_nd.unsupported_dtypes = ('float16', 'uint16', 'uint32', 'uint64',)

# noinspection PyShadowingNames
def gather(
    params: torch.Tensor,
    indices: torch.Tensor,
    axis: Optional[int] = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.gather(params, axis, indices.type(torch.int64))


# noinspection PyShadowingNames
def gather_nd(
    params: torch.Tensor, indices: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    indices_shape = indices.shape
    params_shape = params.shape
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


def multiprocessing(context=None):
    import torch.multiprocessing

    if context is None:
        return torch.multiprocessing
    return torch.multiprocessing.get_context(context)


def indices_where(
    x: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    where_x = torch.where(x)
    res = torch.cat([torch.unsqueeze(item, -1) for item in where_x], -1, out=out)
    return res


indices_where.support_native_out = True


# noinspection PyUnresolvedReferences,PyShadowingNames
def one_hot(
    indices: torch.Tensor,
    depth: int,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.type(torch.int64), depth).to(device)


def shape(x: torch.Tensor, as_array: bool = False) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(x.shape)
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x, as_tensor=False) -> Union[torch.Tensor, int]:
    return torch.tensor(len(x.shape)) if as_tensor else len(x.shape)


def current_backend_str():
    return "torch"
