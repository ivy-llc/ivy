"""
Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

# local
from ivy.core.container import Container

variable = lambda x: _tf.Variable(x, trainable=True)


def execute_with_gradients(func, xs):
    with _tf.GradientTape() as tape:
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    grads = Container(tape.gradient(y, xs))
    return (y, grads, *rest)


def gradient_descent_update(ws, dcdws, lr):
    ws.map(lambda w, key_chain: w.assign(w - (dcdws if key_chain == '' else dcdws.at_key_chain(key_chain)) * lr))
    return ws


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7):
    step = _tf.cast(step, _tf.float32)
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    ws.map(lambda w, kc: w.assign(w - alpha * mw.at_key_chain(kc) / (vw.at_key_chain(kc) ** 0.5 + epsilon)))
    return ws, mw, vw


stop_gradient = _tf.stop_gradient
