"""Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from typing import Union, Optional, Callable

# local
import ivy
from ivy.functional.ivy.gradients import (
    _get_native_arrays_and_indices,
    _zero_gradients_to_none_and_to_ivy,
    _stop_grad_and_index,
)


def variable(x):
    with tf.device(ivy.dev(x, as_native=True)):
        return tf.Variable(x, trainable=True)


def is_variable(x, exclusive=False):
    return isinstance(x, tf.Variable)


def variable_data(x):
    return x.value()


def execute_with_gradients(func, xs, /, *, retain_grads=False, grad_idxs=None):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        xs = ivy.to_native(xs)
        tape.watch(xs)
        func_ret = func(xs)
    arr_idxs, arr_values = _get_native_arrays_and_indices(func_ret)

    if len(arr_values) == 1:
        y = arr_values[0]
    else:
        y = arr_values

    def grad_func(y):
        ret = tape.gradient(y, xs)
        return ret

    if isinstance(y, ivy.NativeArray):
        grads = ivy.to_ivy(grad_func(y))
    else:
        grads_ = ivy.nested_map(y, grad_func, include_derived=True)
        grads = {arr_idxs[i]: grad for i, grad in enumerate(grads_)}

    grads = _zero_gradients_to_none_and_to_ivy(grads)
    grads = _stop_grad_and_index(y, retain_grads, grads, grad_idxs)
    return func_ret, grads


def value_and_grad(func):
    def grad_fn(xs):
        grads = ivy.nested_map(xs, lambda x: ivy.zeros_like(x), include_derived=True)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
            tape.watch(xs)
            y = func(xs)
        y = y.to_native(y)
        grads_ = tape.gradient(y, xs)
        grads_ = ivy.nested_map(
            grads_,
            lambda x: ivy.to_ivy(x),
            include_derived=True,
        )
        grads = _zero_gradients_to_none_and_to_ivy(xs, grads_)
        grad_idxs = ivy.nested_argwhere(grads_, lambda x: ivy.is_ivy_array(x))
        grad_array_vals = list(ivy.multi_index_nest(grads_, grad_idxs))
        xs = ivy.to_ivy(xs)
        if isinstance(xs, ivy.Array):
            grads = grads_
        else:
            ivy.set_nest_at_indices(grads, grad_idxs, grad_array_vals)
        y = ivy.to_ivy(y)
        return y, grads

    return grad_fn


def stop_gradient(
    x: Union[tf.Tensor, tf.Variable],
    preserve_type: bool = True,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_var = is_variable(x)
    x = tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(x_in)
            tape.watch(x_in)
            y = grad_fn(x_in)
        return ivy.to_ivy(tape.jacobian(y, x_in))

    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(ivy.array(x_in))
            tape.watch(x_in)
            y = grad_fn(x_in)
        return _zero_gradients_to_none_and_to_ivy(ivy.to_ivy(tape.gradient(y, x_in)))

    return callback_fn
