"""Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import jax
import jax.lax as jlax
import jaxlib
from jaxlib.xla_extension import Buffer
from ivy.functional.backends.jax import JaxArray
from typing import Optional, Callable


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


# ToDo: modify these functions to track whether variable() has been called
def variable(x, /):
    return x


def is_variable(x, /, *, exclusive=False):
    if exclusive:
        return False
    return isinstance(
        x, (jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer)
    )


def variable_data(x, /):
    return x


def _forward_fn(
    xs, x, func, duplicate_index_chains, xs_grad_idxs=None, ret_grad_idxs=None
):
    x_arr_idxs = ivy.nested_argwhere(x, ivy.is_array)
    x_arr_values = ivy.multi_index_nest(x, x_arr_idxs)
    if xs_grad_idxs is not None:
        ivy.set_nest_at_indices(xs, xs_grad_idxs, x_arr_values)
    elif ivy.is_array(xs):
        xs = x
    else:
        xs_arr_idxs = ivy.nested_argwhere(xs, lambda x: ivy.is_array(x))
        ivy.set_nest_at_indices(xs, xs_arr_idxs, x_arr_values)
    if not ivy.is_array(xs):
        xs = _set_duplicates(xs, duplicate_index_chains)
    ret = func(xs)
    _, ret_values = _get_native_variables_and_indices(ret, idxs=ret_grad_idxs)
    if isinstance(ret_values, list) and len(ret_values) == 1 and ret_grad_idxs is None:
        ret_values = ret_values[0]
    return ret_values


def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    duplicate_index_chains = ()
    if isinstance(xs, ivy.Container):
        duplicate_index_chains = xs.duplicate_array_keychains()
    elif isinstance(xs, (list, tuple, dict)):
        duplicate_index_chains = ivy.duplicate_array_index_chains(xs)
    xs = _arrays_to_float_variables(xs, xs_grad_idxs=xs_grad_idxs)
    if not ivy.is_array(xs):
        xs = _set_duplicates(xs, duplicate_index_chains)
    func_ret = func(xs)
    xs_required = _get_required_native_variables(xs, xs_grad_idxs)
    required_duplicate_index_chains = ()
    if isinstance(xs_required, ivy.Container):
        required_duplicate_index_chains = xs_required.duplicate_array_keychains()
    elif isinstance(xs_required, (list, tuple, dict)):
        required_duplicate_index_chains = ivy.duplicate_array_index_chains(xs_required)
    xs = ivy.to_native(xs)
    ret_idxs, ret_values = _get_native_variables_and_indices(
        func_ret, idxs=ret_grad_idxs
    )
    if ret_values is None or (isinstance(ret_values, list) and len(ret_values) == 0):
        return func_ret, {}
    if isinstance(ret_values, list) and len(ret_values) == 1 and ret_grad_idxs is None:
        y = ret_values[0]
    else:
        y = ret_values
    if isinstance(y, ivy.NativeArray):
        grad_fn = jax.grad(
            lambda x: _forward_fn(
                xs,
                x,
                func,
                duplicate_index_chains,
                xs_grad_idxs=xs_grad_idxs,
                ret_grad_idxs=ret_grad_idxs,
            )
        )
        grads = _set_duplicates(grad_fn(xs_required), required_duplicate_index_chains)
    else:
        grad_fn = jax.jacrev(
            lambda x: _forward_fn(
                xs,
                x,
                func,
                duplicate_index_chains,
                xs_grad_idxs=xs_grad_idxs,
                ret_grad_idxs=ret_grad_idxs,
            )
        )
        grads_ = grad_fn(xs_required)
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
        xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
        value, grad = jax.value_and_grad(grad_fn)(xs)
        grad = _remove_zeros_and_nones(grad, grad)
        return ivy.to_ivy(value), ivy.to_ivy(grad)

    return callback_fn


def stop_gradient(
    x: JaxArray, /, *, preserve_type: bool = True, out: Optional[JaxArray] = None
) -> JaxArray:
    return jlax.stop_gradient(x)


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(jax.jacfwd(grad_fn)((ivy.to_native(x_in))))
    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(jax.grad(grad_fn)(ivy.to_native(x_in)))
    return callback_fn
