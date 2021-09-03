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


def is_variable(x):
    return isinstance(x, _tf.Variable)


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
    with _tf.GradientTape(persistent=retain_grads, watch_accessed_variables=False) as tape:
        tape.watch(xs)
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    grads = Container(tape.gradient(y, xs))
    return y, grads, *rest


def _adam_update_inplace(ws, dcdws, alpha, mw, vw, epsilon):
    ws.map(lambda w, kc: w.assign(w - alpha * mw.at_key_chain(kc) / (vw.at_key_chain(kc) ** 0.5 + epsilon)))
    return ws, mw, vw


def _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon):
    ws = ws.map(lambda w, key_chain: w - alpha * mw.at_key_chain(key_chain) /
                                     (vw.at_key_chain(key_chain) ** 0.5 + epsilon))
    return ws


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=True, stop_gradients=True):
    step = _tf.cast(step, _tf.float32)
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    if inplace:
        return _adam_update_inplace(ws, dcdws, alpha, mw, vw, epsilon)
    if stop_gradients:
        dcdws.stop_gradients(preserve_type=True)
    return _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon)


def stop_gradient(x, preserve_type=True):
    is_var = is_variable(x)
    x = _tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x
