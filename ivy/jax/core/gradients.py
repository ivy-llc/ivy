"""
Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.lax as _jlax

variable = lambda x: x


def execute_with_gradients(func, xs):
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
        grad_fn = lambda x_in: func(x_in)[0]
    else:
        y = func_ret
        rest = tuple()
        grad_fn = func
    grads = _jax.grad(grad_fn)(xs)
    return (y, grads, *rest)


gradient_descent_update = lambda ws, dcdws, lr: [w - dcdw * lr for w, dcdw in zip(ws, dcdws)]
stop_gradient = _jlax.stop_gradient
