# global
import inspect
from typing import Callable

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
def apply_along_axis(func1d: Callable, axis: int, arr, *args, **kwargs):
    ndim = ivy.get_num_dims(arr)
    if axis is None:
        raise ValueError("Axis must be an integer.")
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis = axis + ndim

    func = lambda elem: func1d(elem, *args, **kwargs)
    # apply from the back (out_axes=-1)
    for i in range(1, ndim - axis):
        func = ivy.vmap(func, in_axes=i, out_axes=-1)
    # apply from the front (out_axes=0)
    for i in range(axis):
        func = ivy.vmap(func, in_axes=0, out_axes=0)

    return ivy.asarray(func(arr))


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
