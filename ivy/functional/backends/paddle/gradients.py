"""Collection of Paddle gradient functions, wrapped to fit Ivy syntax and signature."""

# global

from typing import Optional, Callable
import paddle
from itertools import chain

# local
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version
from ivy.functional.ivy.gradients import (
    _get_required_float_variables,
    _get_y_and_ret_idxs,
    _get_native_y,
    _set_duplicates,
    _process_func_ret_and_grads,
)


def variable(x, /):
    if ivy.is_int_dtype(x.dtype):
        x = ivy.astype(x, ivy.default_float_dtype()).to_native()
    if not x.is_leaf:
        ret = x.detach()
        ret.stop_gradient = False
        return ret
    ret = x.clone()
    ret.stop_gradient = False
    return ret


def is_variable(x, /, *, exclusive: bool = False):
    return isinstance(x, paddle.Tensor) and not x.stop_gradient


def variable_data(x: paddle.Tensor, /) -> paddle.Tensor:
    return x.value()


def _grad_func(y, xs, retain_grads):
    """Gradient calculation function."""
    # Creating a zero gradient nest for the case where no gradients are computed
    grads_ = ivy.nested_map(
        xs,
        lambda x: (
            ivy.array([0.0]).to_native()
            if x is None
            else ivy.to_native(ivy.zeros_like(x))
        ),
        include_derived=True,
        shallow=False,
    )

    # Gradient calculation
    if isinstance(xs, paddle.Tensor):
        grads = paddle.grad(
            outputs=y,
            inputs=xs,
            retain_graph=True,
            create_graph=retain_grads,
            allow_unused=True,
        )[0]
        grads = grads_ if grads is None else grads
    elif isinstance(xs, ivy.Container):
        grads = xs.cont_from_flat_list(
            list(
                paddle.grad(
                    outputs=[y],
                    inputs=[
                        ivy.array([0.0]).to_native() if v is None else v
                        for k, v in xs.cont_to_iterator()
                    ],
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
            grads = ivy.add(grads, grads_)
        else:
            grads = grads_ if grads is None else grads
    else:

        def grad_(x):
            x = ivy.array([0.0]).to_native() if x is None else x
            grad = paddle.grad(
                outputs=y,
                inputs=ivy.array([0.0]).to_native() if x is None else x,
                retain_graph=True,
                create_graph=retain_grads,
                allow_unused=True,
            )[0]
            return grad if grad is not None else ivy.to_native(ivy.zeros_like(x))

        grads = ivy.nested_map(xs, grad_, include_derived=True, shallow=False)
        grads = ivy.nested_multi_map(
            lambda x, _: (ivy.add(x[0], x[1])), [grads, grads_]
        )
    return grads


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "float16")}}, backend_version
)
def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    # Conversion of required arrays to float variables and duplicate index chains
    xs, xs1, required_duplicate_index_chains, _ = _get_required_float_variables(
        xs, xs_grad_idxs
    )
    func_ret = func(xs)
    xs = xs1
    if isinstance(xs, ivy.Container):
        duplicate_indices = list(
            chain.from_iterable(
                [
                    map(lambda x: x.split("/"), duplicate_index_chain[1:])
                    for duplicate_index_chain in required_duplicate_index_chains
                ]
            )
        )
        xs = ivy.set_nest_at_indices(xs, duplicate_indices, None, shallow=False)

    # Getting the relevant outputs from the function return for gradient calculation
    y, ret_idxs = _get_y_and_ret_idxs(func_ret, ret_grad_idxs, create_var=True)

    if isinstance(y, ivy.NativeArray):
        # Gradient calculation for a single output
        grads = _set_duplicates(
            _grad_func(paddle.clone(y), xs, retain_grads),
            required_duplicate_index_chains,
        )
    else:
        # Gradient calculation for multiple outputs
        #
        y = _get_native_y(y)
        grad_arr_idxs = ivy.nested_argwhere(y, lambda x: ivy.is_native_array(x))
        grad_arr_values = ivy.multi_index_nest(y, grad_arr_idxs)
        grads_ = [
            _grad_func(paddle.clone(arr_value), xs, retain_grads)
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
            grad = paddle.grad(y, x, allow_unused=True)[0]
            grad = (
                grad
                if grad is not None
                else ivy.to_native(ivy.zeros_like(ivy.to_ivy(x)))
            )
            grad = ivy.to_ivy(grad)
            return grad

        grads = ivy.nested_map(xs, autograd_fn, include_derived=True, shallow=False)
        y = ivy.to_ivy(y)
        return y, grads

    return callback_fn


def stop_gradient(
    x: Optional[paddle.Tensor],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[paddle.Tensor] = None,
):
    is_var = is_variable(x)
    x.stop_gradient = True
    if is_var and preserve_type:
        return variable(x)
    return x


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(
        paddle.incubate.autograd.Jacobian(grad_fn, ivy.to_native(x_in))
    )
    return callback_fn


def grad(f, argnums=0):
    if grad.nth == 0:
        grad.f_original = f

    # ToDo: Return grads on nth chained calls rather than None. issue with paddle.grad.
    def _nth_derivative(n):
        def _inner(x):
            x = ivy.to_native(x)
            if n == 0:
                x.stop_gradient = False
                ret = grad.f_original(x) if grad.f_original is not None else f(x)
                grad.nth = 0
                return ret
            else:
                x.stop_gradient = False
                y = _nth_derivative(n - 1)(x)
                y = ivy.to_native(y)
                y_ones = paddle.ones_like(y)
                y_ones.stop_gradient = False
                y.stop_gradient = False
                dy_dx = paddle.grad(
                    outputs=[y],
                    inputs=[x],
                    create_graph=True,
                    grad_outputs=y_ones,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            return dy_dx

        return _inner

    grad.nth += 1

    return _nth_derivative(grad.nth)


grad.f_original = None
grad.nth = 0
