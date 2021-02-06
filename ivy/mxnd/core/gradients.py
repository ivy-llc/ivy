"""
Collection of MXNet gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx


def variable(array_in):
    array_in.attach_grad()
    return array_in


# noinspection PyUnresolvedReferences
def execute_with_gradients(func, xs):
    with _mx.autograd.record():
        func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    _mx.autograd.backward(y)
    return (y, [x.grad for x in xs], *rest)


def gradient_descent_update(ws, dcdws, lr):
    ws = [w - dcdw * lr for w, dcdw in zip(ws, dcdws)]
    [w.attach_grad() for w in ws]
    return ws


stop_gradient = _mx.nd.stop_gradient
