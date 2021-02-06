"""
Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

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
    grads = tape.gradient(y, xs)
    return (y, grads, *rest)


def gradient_descent_update(ws, dcdws, lr):
    [w.assign(w - dcdw * lr) for w, dcdw in zip(ws, dcdws)]
    return ws


stop_gradient = _tf.stop_gradient
