"""Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
import warnings
from typing import Optional, Callable
import numpy as np

# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_native_arrays_and_indices,
    _zero_gradients_to_none_and_to_ivy,
    _stop_grad_and_index,
)


def variable(x):
    if not x.is_leaf:
        return x.detach().requires_grad_()
    return x.clone().requires_grad_()


def is_variable(x, /, *, exclusive: bool = False):
    return isinstance(x, torch.Tensor) and x.requires_grad


def variable_data(x):
    return x.data


def _forward_fn(xs, func):
    xs = ivy.Container(xs)
    print("xs", xs)
    ret = func(xs)

    if isinstance(ret, ivy.Array):
        array_values = ret.to_native()
    else:
        array_idxs = ivy.nested_argwhere(ret, lambda x: ivy.is_native_array(x))
        if (
            not isinstance(array_idxs, list)
            or np.asarray(array_idxs, "object").size == 0
        ):
            array_values = []
        else:
            array_values = ivy.multi_index_nest(ret, array_idxs)

    return array_values


# noinspection PyShadowingNames
def execute_with_gradients(func, xs, /, *, retain_grads=False, grad_idxs=None):
    func_ret = func(xs)
    xs = ivy.to_native(xs)
    arr_idxs, arr_values = _get_native_arrays_and_indices(func_ret)

    if arr_values is None or (isinstance(arr_values, list) and len(arr_values) == 0):
        return func_ret, {}
    if isinstance(arr_values, list) and len(arr_values) == 1:
        y = arr_values[0]
    else:
        y = arr_values

    def grad_func(y):
        if isinstance(xs, ivy.Container):
            grads = xs.from_flat_list(
                list(
                    torch.autograd.grad(
                        [y],
                        [v for k, v in xs.to_iterator()],
                        retain_graph=True,
                        create_graph=retain_grads,
                        allow_unused=True,
                    )
                )
            )
        else:
            grads = torch.autograd.grad(
                y,
                xs,
                retain_graph=True,
                create_graph=retain_grads,
                allow_unused=True,
            )[0]
        return grads

    if isinstance(y, ivy.NativeArray):
        grads = grad_func(torch.clone(y))
    else:
        # ToDo: use functorch.jacrev if it fixes the issue with broken memory reference
        array_idxs = ivy.nested_argwhere(y, lambda x: ivy.is_native_array(x))
        if (
            not isinstance(array_idxs, list)
            or np.asarray(array_idxs, "object").size == 0
        ):
            y = []
        else:
            y = ivy.multi_index_nest(y, array_idxs)

        grad_arr_idxs = ivy.nested_argwhere(y, lambda x: ivy.is_native_array(x))
        grad_arr_values = ivy.multi_index_nest(y, grad_arr_idxs)
        grads_ = [grad_func(torch.clone(arr_value)) for arr_value in grad_arr_values]
        grads = grads_
        if isinstance(arr_idxs, list) and len(arr_idxs):
            grads = {arr_idxs[i]: grad for i, grad in enumerate(grads_)}

    grads = _zero_gradients_to_none_and_to_ivy(grads)
    func_ret, grads = _stop_grad_and_index(func_ret, retain_grads, grads, grad_idxs)
    return func_ret, grads


def value_and_grad(func):
    grad_fn = lambda xs: ivy.to_native(func(xs))

    def callback_fn(xs):
        y = grad_fn(xs)

        def autograd_fn(x):
            x = ivy.to_native(x)
            grad = torch.autograd.grad(y, x, allow_unused=True)[0]
            grad = (
                grad
                if grad is not None
                else ivy.to_native(ivy.zeros_like(ivy.to_ivy(x)))
            )
            grad = ivy.to_ivy(grad)
            grad = _zero_gradients_to_none_and_to_ivy(grad)
            return grad

        grads = ivy.nested_map(
            xs,
            autograd_fn,
            include_derived=True,
        )
        y = ivy.to_ivy(y)
        return y, grads

    return callback_fn


def stop_gradient(
    x: Optional[torch.Tensor],
    preserve_type: bool = True,
    *,
    out: Optional[torch.Tensor] = None,
):
    if is_variable(x) and preserve_type:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if x.grad_fn:
                x = x.detach()
                x.requires_grad = True
            elif x.grad:
                x.grad.data.zero_()
        return x
    return x.detach()


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(
        torch.autograd.functional.jacobian(grad_fn, ivy.to_native(x_in))
    )
    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        x = ivy.to_native(ivy.array(x_in)).detach()
        x.requires_grad = True
        grad_fn(x).backward()
        return ivy.to_ivy(x.grad)

    return callback_fn
