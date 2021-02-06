"""
Collection of Numpy gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import logging


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


gradient_descent_update = lambda ws, dcdws, lr: [w - dcdw * lr for w, dcdw in zip(ws, dcdws)]


def stop_gradient(array_in):
    logging.warning('Numpy does not support autograd, '
                    '"stop_gradient" has no effect on the array, as gradients are not supported in the first place.')
    return array_in
