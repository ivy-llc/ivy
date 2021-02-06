"""
Collection of PyTorch reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


def reduce_sum(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _torch.sum(x, dim=axis, keepdim=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    if isinstance(axis, int):
        return _torch.prod(x, dim=axis, keepdim=keepdims)
    for i, a in enumerate(axis):
        x = _torch.prod(x, dim=a if keepdims else a - i, keepdim=keepdims)
    return x


def reduce_mean(x, axis=None, keepdims=False):
    if axis is None:
        num_dims = len(x.shape)
        axis = tuple(range(num_dims))
    elif isinstance(axis, list):
        axis = tuple(axis)
    return _torch.mean(x, dim=axis, keepdim=keepdims)


def reduce_min(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        axis = [axis]
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    for i, a in enumerate(axis):
        x = _torch.min(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    return x


def reduce_max(x, axis=None, num_x_dims=None, keepdims=False):
    if num_x_dims is None:
        num_x_dims = len(x.shape)
    if axis is None:
        axis = list(range(num_x_dims))
    elif isinstance(axis, int):
        return _torch.max(x, dim=axis, keepdim=keepdims).values
    axis = [(item + num_x_dims) % num_x_dims for item in axis]  # prevent negative indices
    axis.sort()
    for i, a in enumerate(axis):
        x = _torch.max(x, dim=a if keepdims else a - i, keepdim=keepdims).values
    return x
