"""
Collection of MXNet gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx


def variable(x):
    x.attach_grad()
    return x


def is_variable(x):
    return isinstance(x, _mx.ndarray.ndarray.NDArray) and x.grad is not None


def inplace_update(x, val):
    x = val
    return x


def inplace_decrement(x, val):
    x -= val
    return x


def inplace_increment(x, val):
    x += val
    return x


# noinspection PyUnresolvedReferences
def execute_with_gradients(func, xs, retain_grads=False):
    xs.map(lambda x, kc: x.attach_grad())
    with _mx.autograd.record():
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    x_grads_flat = _mx.autograd.grad(y, [v for k, v in xs.to_iterator()], retain_graph=retain_grads,
                                     create_graph=retain_grads)
    return y, xs.from_flat_list(x_grads_flat), *rest


def stop_gradient(x, preserve_type=True):
    is_var = is_variable(x)
    x = _mx.nd.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x
