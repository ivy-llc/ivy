# global
import inspect

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def diagonal(a, offset=0, axis1=0, axis2=1):
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@to_ivy_arrays_and_back
def diag(v, k=0):
    return ivy.diag(v, k=k)


@to_ivy_arrays_and_back
def diag_indices(n, ndim=2):
    idx = ivy.arange(n, dtype=int)
    return (idx,) * ndim


# take_along_axis
@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis, mode="fill"):
    return ivy.take_along_axis(arr, indices, axis, mode=mode)


@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    return ivy.tril_indices(n, m, k)


@to_ivy_arrays_and_back
def triu_indices(n, k=0, m=None):
    return ivy.triu_indices(n, m, k)


@to_ivy_arrays_and_back
def triu_indices_from(arr, k=0):
    return ivy.triu_indices(arr.shape[-2], arr.shape[-1], k)


@to_ivy_arrays_and_back
def tril_indices_from(arr, k=0):
    return ivy.tril_indices(arr.shape[-2], arr.shape[-1], k)


# unravel_index
@to_ivy_arrays_and_back
def unravel_index(indices, shape):
    ret = [x.astype(indices.dtype) for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)


@to_ivy_arrays_and_back
def mask_indices(n, mask_func, k=0):
    mask_func_obj = inspect.unwrap(mask_func)
    mask_func_name = mask_func_obj.__name__
    try:
        ivy_mask_func_obj = getattr(ivy.functional.frontends.jax.numpy, mask_func_name)
        a = ivy.ones((n, n))
        mask = ivy_mask_func_obj(a, k=k)
        indices = ivy.argwhere(mask.ivy_array)
        return indices[:, 0], indices[:, 1]
    except AttributeError as e:
        print(f"Attribute error: {e}")


@to_ivy_arrays_and_back
def diag_indices_from(arr):
    print(arr)
    n = arr.shape[0]
    ndim = ivy.get_num_dims(arr)
    if not all(arr.shape[i] == n for i in range(ndim)):
        raise ValueError("All dimensions of input must be of equal length")
    idx = ivy.arange(n, dtype=int)
    return (idx,) * ndim


@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    if sparse:
        return tuple(
            ivy.arange(dim)
            .expand_dims(
                axis=[j for j in range(len(dimensions)) if i != j],
            )
            .astype(dtype)
            for i, dim in enumerate(dimensions)
        )
    else:
        grid = ivy.meshgrid(*[ivy.arange(dim) for dim in dimensions], indexing="ij")
        return ivy.stack(grid, axis=0).astype(dtype)

@to_ivy_arrays_and_back
def take(arr, indices, axis=None, out=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None):
    return ivy.take(arr, indices, axis=axis, out=out, mode=mode, unique_indices=unique_indices, indices_are_sorted=indices_are_sorted, fill_value=fill_value)

@to_ivy_arrays_and_back
def choose(arr, choices, out=None, mode='raise'):
    if mode not in ['raise', 'wrap', 'clip']:
        raise ValueError("mode must be 'raise', 'wrap', or 'clip'")
    if out is None:
        out = ivy.zeros(arr.shape, dtype=choices.dtype)
    for i, index in enumerate(arr):
        if mode == 'raise' and (index < 0 or index >= len(choices)):
            raise ValueError("invalid entry in choice array")
        if mode == 'wrap':
            index = index % len(choices)
        elif mode == 'clip':
            index = min(max(index, 0), len(choices)-1)
        out[i] = choices[index][i]
    return out