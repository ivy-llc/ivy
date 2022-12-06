"""Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
import warnings
from typing import Optional, Callable
import numpy as np

# local
import ivy
from ivy.functional.ivy.gradients import (
    _arrays_to_float_variables,
    _get_required_native_variables,
    _get_native_variables_and_indices,
    _remove_zeros_and_nones,
    _set_duplicates,
    _stop_grad_and_index,
)


def variable(x, /):
    if not x.is_leaf:
        return x.detach().requires_grad_()
    return x.clone().requires_grad_()


def is_variable(x, /, *, exclusive: bool = False):
    return isinstance(x, torch.Tensor) and x.requires_grad


def variable_data(x, /):
    return x.data


# noinspection PyShadowingNames
def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    duplicate_index_chains = ()
    if isinstance(xs, ivy.Container):
        duplicate_index_chains = xs.duplicate_array_keychains()
    elif isinstance(xs, (list, tuple, dict)):
        duplicate_index_chains = ivy.duplicate_array_index_chains(xs)
    xs = _arrays_to_float_variables(xs, xs_grad_idxs=xs_grad_idxs)
    xs = _set_duplicates(xs, duplicate_index_chains)
    func_ret = func(xs)
    xs = _get_required_native_variables(xs, xs_grad_idxs)
    required_duplicate_index_chains = ()
    if isinstance(xs, ivy.Container):
        required_duplicate_index_chains = xs.duplicate_array_keychains()
    elif isinstance(xs, (list, tuple, dict)):
        required_duplicate_index_chains = ivy.duplicate_array_index_chains(xs)
    ret_idxs, ret_values = _get_native_variables_and_indices(
        func_ret,
        idxs=ret_grad_idxs,
        create_var=True,
    )
    if ret_values is None or (isinstance(ret_values, list) and len(ret_values) == 0):
        return func_ret, {}
    if isinstance(ret_values, list) and len(ret_values) == 1 and ret_grad_idxs is None:
        y = ret_values[0]
    else:
        y = ret_values

    def grad_func(y):
        grads_ = ivy.nested_map(
            xs, lambda x: ivy.to_native(ivy.zeros_like(x)), include_derived=True
        )
        if isinstance(xs, ivy.NativeArray):
            grads = torch.autograd.grad(
                y,
                xs,
                retain_graph=True,
                create_graph=retain_grads,
                allow_unused=True,
            )[0]
            grads = grads_ if grads is None else grads
        elif isinstance(xs, ivy.Container):
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
            if isinstance(grads, ivy.Container):
                grads = ivy.nested_map(
                    grads, lambda x: 0 if x is None else x, include_derived=True
                )
                grads += grads_
            else:
                grads = grads_ if grads is None else grads
        else:

            def grad_(x):
                grad = torch.autograd.grad(
                    y,
                    x,
                    retain_graph=True,
                    create_graph=retain_grads,
                    allow_unused=True,
                )[0]
                return grad if grad is not None else 0

            grads = ivy.nested_map(
                xs,
                grad_,
                include_derived=True,
            )
            grads = ivy.nested_multi_map(lambda x, _: (x[0] + x[1]), [grads, grads_])
        return grads

    if isinstance(y, ivy.NativeArray):
        grads = _set_duplicates(
            grad_func(torch.clone(y)), required_duplicate_index_chains
        )
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
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {
                ret_idxs[i]: _set_duplicates(grad, required_duplicate_index_chains)
                for i, grad in enumerate(grads_)
            }
    grads = ivy.nested_map(
        grads,
        lambda x: ivy.where(ivy.isfinite(x), x, 0) if ivy.is_array(x) else x,
        include_derived=True,
    )
    func_ret, grads = _stop_grad_and_index(func_ret, retain_grads, grads)
    grads = ivy.to_ivy(grads)
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
            grad = _remove_zeros_and_nones(grad, grad)
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
    /,
    *,
    preserve_type: bool = True,
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
