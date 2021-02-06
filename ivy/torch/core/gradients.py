"""
Collection of PyTorch gradient functions, wrapped to fit Ivy syntax and signature.
"""


def variable(array_in):
    array_in.requires_grad = True
    return array_in


def execute_with_gradients(func, xs):
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    y.backward()
    return (y, [x.grad for x in xs], *rest)


def gradient_descent_update(ws, dcdws, lr):
    ws = [w - dcdw * lr for w, dcdw in zip(ws, dcdws)]
    [w.retain_grad() for w in ws]
    return ws


def stop_gradient(x):
    x.requires_grad = False
    return x
