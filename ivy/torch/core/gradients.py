"""
Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch
import warnings as _warnings


def variable(x):
    if not x.is_leaf:
        return x.detach().requires_grad_()
    return x.clone().requires_grad_()


def is_variable(x):
    return isinstance(x, _torch.Tensor) and x.requires_grad


def inplace_update(x, val):
    x.data = val
    return x


def execute_with_gradients(func, xs, retain_grads=False):
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    x_grads_flat = list(_torch.autograd.grad([y], [v for k, v in xs.to_iterator()], retain_graph=retain_grads,
                                             create_graph=retain_grads))
    return (y, xs.from_flat_list(x_grads_flat), *rest)


def _gradient_descent_update_trackable(ws, dcdws, lr):
    return ws.map(lambda w, key_chain: (w - dcdws.at_key_chain(key_chain) * lr))


def _gradient_descent_update_inplace(ws, dcdws, lr):

    def _inplace_update(x, key_chain):
        x.data -= dcdws.at_key_chain(key_chain) * lr

    ws.map(_inplace_update)
    return ws


def gradient_descent_update(ws, dcdws, lr, inplace=True, stop_gradients=True):
    if inplace:
        ret = _gradient_descent_update_inplace(ws, dcdws, lr)
    else:
        ret = _gradient_descent_update_trackable(ws, dcdws, lr)
    if stop_gradients:
        dcdws.stop_gradients(preserve_type=True)
    return ret


def _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon):
    ws = ws.map(lambda w, key_chain: (w - alpha * mw.at_key_chain(key_chain) /
                                      (vw.at_key_chain(key_chain) ** 0.5 + epsilon)))
    return ws, mw, vw


def _adam_update_inplace(ws, dcdws, alpha, mw, vw, epsilon):

    def _inplace_update(x, key_chain):
        x.data -= alpha * mw.at_key_chain(key_chain) / (vw.at_key_chain(key_chain) ** 0.5 + epsilon)

    ws.map(_inplace_update)
    return ws, mw, vw


def adam_update(ws, dcdws, lr, mw, vw, step, beta1=0.9, beta2=0.999, epsilon=1e-7, inplace=True, stop_gradients=True):
    step = step.type(_torch.float32)
    mw = dcdws.map(lambda dcdw, kc: beta1 * mw.at_key_chain(kc) + (1 - beta1) * dcdw)
    dcdws_sqrd = dcdws.map(lambda dcdw, _: dcdw ** 2)
    vw = dcdws_sqrd.map(lambda dcdw_sqrd, kc: beta2 * vw.at_key_chain(kc) + (1 - beta2) * dcdw_sqrd)
    beta1_pow = beta1 ** step
    beta2_pow = beta2 ** step
    alpha = lr * (1 - beta2_pow)**0.5 / (1 - beta1_pow + epsilon)

    if inplace:
        ret = _adam_update_inplace(ws, dcdws, alpha, mw, vw, epsilon)
    else:
        ret = _adam_update_trackable(ws, dcdws, alpha, mw, vw, epsilon)
    if stop_gradients:
        dcdws.stop_gradients(preserve_type=True)
    return ret


def stop_gradient(x, preserve_type=True):
    if is_variable(x) and preserve_type:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            if x.grad:
                x.grad.data.zero_()
        return x
    return x.detach()
