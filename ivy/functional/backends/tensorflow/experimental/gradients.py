# global
import tensorflow as tf
from typing import Callable

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import _get_required_float_variables


def bind_custom_gradient_function(func, custom_grad_fn):
    @tf.custom_gradient
    def custom_module(x):
        x, _, _, _, _ = _get_required_float_variables(x, xs_grad_idxs=None)
        ret = func(x)

        def grad(upstream):
            return custom_grad_fn((x, ret), upstream)

        return ivy.to_native((ret, grad), nested=True, include_derived=True)

    return inputs_to_native_arrays(custom_module)


def vjp(func: Callable, *primals):
    def grad_fn(*x_in):
        return ivy.to_native(
            func(*ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(ivy.to_native(primals, nested=True))
        primals_out = grad_fn(*ivy.to_native(primals, nested=True))

    def vjpfun(x_in):
        grads = tape.gradient(
            primals_out,
            ivy.to_native(primals, nested=True),
            output_gradients=ivy.to_native(x_in, nested=True),
        )
        return ivy.to_ivy(grads, nested=True, include_derived=True)

    return (ivy.to_ivy(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    def grad_fn(x_in):
        return ivy.to_native(
            func(ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    primals = ivy.to_native(primals, nested=True)
    tangents = ivy.to_native(tangents, nested=True)

    with tf.autodiff.ForwardAccumulator(
        primals,
        tangents,
    ) as acc:
        primals_out = grad_fn(*primals)
    tangents_out = acc.jvp(primals_out)

    return ivy.to_ivy((primals_out, tangents_out), nested=True, include_derived=True)
