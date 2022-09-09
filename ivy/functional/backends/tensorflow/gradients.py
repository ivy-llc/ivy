"""Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from typing import Union, Optional

# local
import ivy


def variable(x):
    with tf.device(ivy.dev(x, as_native=True)):
        return tf.Variable(x, trainable=True)


def is_variable(x, exclusive=False):
    return isinstance(x, tf.Variable)


def variable_data(x):
    return x.value()


def execute_with_gradients(func, xs, retain_grads=False):
    with tf.GradientTape(
        persistent=retain_grads, watch_accessed_variables=False
    ) as tape:
        tape.watch(xs)
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    y = ivy.to_native(y)
    grads = tape.gradient(y, xs)
    y = ivy.to_ivy(y)
    if not retain_grads:
        y = ivy.stop_gradient(y)
    return (y, grads, *rest)


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
        grad_idxs = ivy.nested_indices_where(grads_, lambda x: x is not None)
        grad_array_vals = list(ivy.multi_index_nest(grads_, grad_idxs))
        ivy.set_nest_at_indices(grads, grad_idxs, grad_array_vals)
        y = ivy.to_ivy(y)
        return y, grads

    return grad_fn


def stop_gradient(
    x: Union[tf.Tensor, tf.Variable],
    preserve_type: bool = True,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None
) -> Union[tf.Tensor, tf.Variable]:
    is_var = is_variable(x)
    x = tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x
