"""Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
import warnings
from typing import Optional, Callable

# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_required_float_variables,
    _get_y_and_ret_idxs,
    _get_native_y,
    _set_duplicates,
    _process_func_ret_and_grads,
)


def variable(x, /):
    if not x.is_leaf:
        return x.detach().requires_grad_()
    return x.clone().requires_grad_()


def is_variable(x, /, *, exclusive: bool = False):
    return isinstance(x, torch.Tensor) and x.requires_grad


def variable_data(x, /):
    return x.data


def _grad_func(y, xs, retain_grads):
    """Gradient calculation function."""
    # Creating a zero gradient nest for the case where no gradients are computed
    grads_ = ivy.nested_map(
        xs, lambda x: ivy.to_native(ivy.zeros_like(x)), include_derived=True
    )

    # Gradient calculation
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
        grads = xs.cont_from_flat_list(
            list(
                torch.autograd.grad(
                    [y],
                    [v for k, v in xs.cont_to_iterator()],
                    retain_graph=True,
                    create_graph=retain_grads,
                    allow_unused=True,
                )
            )
        )
        # Returning zeros if no gradients are computed for consistent results
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


def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    # Conversion of required arrays to float variables and duplicate index chains
    xs, xs1, required_duplicate_index_chains, _ = _get_required_float_variables(
        xs, xs_grad_idxs
    )

    func_ret = func(xs)
    xs = xs1

    # Getting the relevant outputs from the function return for gradient calculation
    y, ret_idxs = _get_y_and_ret_idxs(func_ret, ret_grad_idxs, create_var=True)

    if isinstance(y, ivy.NativeArray):
        # Gradient calculation for a single output
        grads = _set_duplicates(
            _grad_func(torch.clone(y), xs, retain_grads),
            required_duplicate_index_chains,
        )
    else:
        # Gradient calculation for multiple outputs
        # ToDo: use functorch.jacrev if it fixes the issue with broken memory reference
        y = _get_native_y(y)
        grad_arr_idxs = ivy.nested_argwhere(y, lambda x: ivy.is_native_array(x))
        grad_arr_values = ivy.multi_index_nest(y, grad_arr_idxs)
        grads_ = [
            _grad_func(torch.clone(arr_value), xs, retain_grads)
            for arr_value in grad_arr_values
        ]
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {
                ret_idxs[i]: _set_duplicates(grad, required_duplicate_index_chains)
                for i, grad in enumerate(grads_)
            }

    # Stop further gradient propagation if not retaining gradients
    return _process_func_ret_and_grads(func_ret, grads, retain_grads)


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
