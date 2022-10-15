"""Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import jax
import jax.lax as jlax
import jaxlib
from jaxlib.xla_extension import Buffer
from ivy.functional.backends.jax import JaxArray
from typing import Optional, Callable
from itertools import chain


# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_native_arrays_and_indices,
    _zero_gradients_to_none_and_to_ivy,
    _stop_grad_and_index,
)


# ToDo: modify these functions to track whether variable() has been called
def variable(x):
    return x


def is_variable(x, exclusive=False):
    if exclusive:
        return False
    return isinstance(
        x, (jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer)
    )


def variable_data(x):
    return x


def _set_duplicates(xs, duplicate_key_chains):
    originals = [
        [key_chains[0]] * (len(key_chains) - 1) for key_chains in duplicate_key_chains
    ]
    originals = ivy.multi_index_nest(xs, list(chain(*originals)))
    duplicates = [list(key_chains[1:]) for key_chains in duplicate_key_chains]
    nullifying_key_chains = [
        keychains.split("/") for keychains in list(chain(*duplicates))
    ]
    ivy.set_nest_at_indices(xs, nullifying_key_chains, originals)
    return xs


def _forward_fn(xs, func, duplicate_key_chains):
    if isinstance(xs, ivy.Container):
        xs = _set_duplicates(xs, duplicate_key_chains)

    ret = func(xs)

    if isinstance(ret, ivy.Array):
        array_values = ret.to_native()
    else:
        ret = ivy.nested_map(ret, lambda x: ivy.to_native(x), include_derived=True)
        array_idxs = ivy.nested_argwhere(ret, lambda x: ivy.is_native_array(x))
        array_values = ivy.multi_index_nest(ret, array_idxs)

    return array_values


def execute_with_gradients(func, xs, /, *, retain_grads=False, grad_idxs=None):
    func_ret = func(xs)
    xs = ivy.to_native(xs)
    arr_idxs, arr_values = _get_native_arrays_and_indices(func_ret)

    if len(arr_values) == 1:
        y = arr_values[0]
    else:
        y = arr_values

    duplicate_key_chains = ()
    if isinstance(xs, ivy.Container):
        duplicate_key_chains = xs.duplicate_array_keychains()

    if isinstance(y, ivy.NativeArray):
        grad_fn = jax.grad(lambda x: _forward_fn(x, func, duplicate_key_chains))
        grads = grad_fn(xs)
    else:
        grad_fn = jax.jacrev(lambda x: _forward_fn(x, func, duplicate_key_chains))
        grads_ = grad_fn(xs)
        grads = {arr_idxs[i]: grad for i, grad in enumerate(grads_)}

    if isinstance(xs, ivy.Container):
        grads = _set_duplicates(grads, duplicate_key_chains)

    grads = _zero_gradients_to_none_and_to_ivy(grads)
    grads = _stop_grad_and_index(y, retain_grads, grads, grad_idxs)
    return func_ret, grads


def value_and_grad(func):
    grad_fn = lambda xs: ivy.to_native(func(xs))

    def callback_fn(xs):
        xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
        ret = jax.value_and_grad(grad_fn)(xs)
        ret = ivy.nested_map(ret, lambda x: ivy.to_ivy(x), include_derived=True)
        return ret

    return callback_fn


def stop_gradient(
    x: JaxArray, preserve_type: bool = True, *, out: Optional[JaxArray] = None
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
