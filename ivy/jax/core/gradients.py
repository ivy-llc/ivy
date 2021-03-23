"""
Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.lax as _jlax
import jax.numpy as _jnp

# local
from ivy.core.container import Container

# ToDo: modify these functions to track whether variable() has been called
variable = lambda x: x
is_variable = lambda x: True


def execute_with_gradients(func, xs):
    xs = xs.to_dict()
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
        grad_fn = lambda x_in: func(x_in)[0]
    else:
        y = func_ret
        rest = tuple()
        grad_fn = func
    grads = Container(_jax.grad(grad_fn)(xs))
    return (y, grads, *rest)


def gradient_descent_update(ws, dcdws, lr):
    ws = ws.map(lambda w, key_chain: (w - (dcdws if key_chain == '' else dcdws.at_key_chain(key_chain)) * lr))
    return ws


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7):
    step = step.astype(_jnp.float32)
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    ws = ws.map(lambda w, kc: w - alpha * mw.at_key_chain(kc) / (vw.at_key_chain(kc) ** 0.5 + epsilon))
    return ws, mw, vw


stop_gradient = _jlax.stop_gradient
