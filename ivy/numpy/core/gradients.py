"""
Collection of Numpy gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import logging
import numpy as _np


def variable(array_in):
    logging.warning('Numpy does not support autograd, '
                    'declaring a "variable" is identical to declaring an "array" when using numpy backend')
    return array_in


def execute_with_gradients(func, xs):
    logging.warning('Numpy does not support autograd, '
                    '"execute_with_gradients" returns None in place of function gradients.')
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    return (y, None, *rest)


def gradient_descent_update(ws, dcdws, lr):
    ws = ws.map(lambda w, key_chain: (w - (dcdws if key_chain == '' else dcdws.at_key_chain(key_chain)) * lr))
    return ws


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7):
    step = step.astype(_np.float32)
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    ws = ws.map(lambda w, kc: w - alpha * mw.at_key_chain(kc) / (vw.at_key_chain(kc) ** 0.5 + epsilon))
    return ws, mw, vw


def stop_gradient(array_in):
    logging.warning('Numpy does not support autograd, '
                    '"stop_gradient" has no effect on the array, as gradients are not supported in the first place.')
    return array_in
