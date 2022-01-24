"""
Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

# local
import ivy
from ivy.core.container import Container


def variable(x):
    with _tf.device('/' + ivy.dev_str(x).upper()):
        return _tf.Variable(x, trainable=True)


def is_variable(x, exclusive=False):
    return isinstance(x, _tf.Variable)


variable_data = lambda x: x.value()


def inplace_update(x, val):
    x.assign(val)
    return x


def inplace_decrement(x, val):
    x.assign(x - val)
    return x


def inplace_increment(x, val):
    x.assign(x + val)
    return x


def execute_with_gradients(func, xs, retain_grads=False):
    if ivy.wrapped_mode():
        xs = xs.to_native()
    with _tf.GradientTape(persistent=retain_grads, watch_accessed_variables=False) as tape:
        tape.watch(xs)
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    if ivy.wrapped_mode():
        y = ivy.to_native(y)
    grads = Container(tape.gradient(y, xs))
    if ivy.wrapped_mode():
        grads = grads.to_ivy()
        y = ivy.to_ivy(y)
    if not retain_grads:
        y = ivy.stop_gradient(y)
    return (y, grads, *rest)


def stop_gradient(x, preserve_type=True):
    is_var = is_variable(x)
    x = _tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x
