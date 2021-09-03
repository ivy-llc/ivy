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


def _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon):
    ws = ws.map(lambda w, key_chain: (w - alpha * mw.at_key_chain(key_chain) /
                                      (vw.at_key_chain(key_chain) ** 0.5 + epsilon)))
    ws.map(lambda w, _: w.attach_grad())
    return ws, mw, vw


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=True, stop_gradients=True):
    # ToDo: see if more efficient in-place method can be implemented
    step = step.reshape((1,)).astype('float32')
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)
    ret = _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon)
    if stop_gradients:
        dcdws.stop_gradients(preserve_type=True)
    return ret


def stop_gradient(x, preserve_type=True):
    is_var = is_variable(x)
    x = _mx.nd.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x
