"""Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
import functorch
import warnings
from typing import Optional, Callable

# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_native_arrays_and_indices,
    _forward_fn,
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


# noinspection PyShadowingNames
def execute_with_gradients(func, xs, /, *, retain_grads=False, grad_idxs=None):
    func_ret = func(xs)
    xs = ivy.to_native(xs)
    arr_idxs, arr_values = _get_native_arrays_and_indices(func_ret)

    if len(arr_values) == 1:
        y = arr_values[0]
    else:
        y = arr_values

    if isinstance(y, ivy.NativeArray):
        if isinstance(xs, ivy.Container):
            grads = xs.from_flat_list(
                list(
                    torch.autograd.grad(
                        [y],
                        [v for k, v in xs.to_iterator()],
                        retain_graph=retain_grads,
                        create_graph=retain_grads,
                        allow_unused=True,
                    )
                )
            )
        else:
            grads = torch.autograd.grad(
                y,
                xs,
                retain_graph=retain_grads,
                create_graph=retain_grads,
                allow_unused=True,
            )[0]
    else:
        if isinstance(xs, ivy.Container):
            xs = xs.to_dict()
        grad_func = functorch.jacrev(lambda x: _forward_fn(x, func))
        grads_ = grad_func(xs)
        if isinstance(xs, dict):
            xs = ivy.Container(**xs)
        grads = {arr_idxs[i]: grad for i, grad in enumerate(grads_)}

    grads = _zero_gradients_to_none_and_to_ivy(grads)
    grads = _stop_grad_and_index(y, retain_grads, grads, grad_idxs)
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
