"""
Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.lax as _jlax
import jaxlib as _jaxlib
from jaxlib.xla_extension import Buffer

# local
import ivy
from ivy.core.container import Container

# ToDo: modify these functions to track whether variable() has been called
variable = lambda x: x
# noinspection PyUnresolvedReferences,PyProtectedMember


def is_variable(x, exclusive=False):
    if exclusive:
        return False
    return isinstance(x, (_jax.interpreters.xla._DeviceArray, _jaxlib.xla_extension.DeviceArray, Buffer))


variable_data = lambda x: x


def inplace_update(x, val):
    return val


def inplace_decrement(x, val):
    x -= val
    return x


def inplace_increment(x, val):
    x += val
    return x


def execute_with_gradients(func, xs, retain_grads=False):
    wrapped_mode = False
    if ivy.wrapped_mode():
        wrapped_mode = True
        # ToDo: find more elegant solution than setting the global wrap mode to false.
        #  There must be some wrongly wrapped jax functions.
        ivy.set_wrapped_mode(False)
        xs = xs.to_native()
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
        grad_fn = lambda x_in: ivy.reshape(func(x_in)[0], [])
    else:
        y = func_ret
        rest = tuple()
        grad_fn = lambda x_in: ivy.reshape(func(x_in), [])
    grad_func = _jax.grad(grad_fn)
    grads = grad_func(xs)
    grads = Container(grads)
    if wrapped_mode:
        # ToDo: find more elegant solution than setting the global wrap mode to false.
        #  There must be some wrongly wrapped jax functions.
        ivy.set_wrapped_mode(True)
        grads = grads.to_ivy()
    if not retain_grads:
        y = ivy.stop_gradient(y)
    return (y, grads, *rest)


stop_gradient = lambda x, preserve_type=True: _jlax.stop_gradient(x)
