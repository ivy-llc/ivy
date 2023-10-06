# global
import jax
from typing import Callable

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays


def bind_custom_gradient_function(func, custom_grad_fn):
    def custom_forward(x):
        ret = func(x)
        return ivy.to_native((ret, (x, ret)), nested=True, include_derived=True)

    def custom_backward(*args):
        return (custom_grad_fn(*args),)

    func = jax.custom_vjp(func)
    func.defvjp(custom_forward, custom_backward)
    return inputs_to_native_arrays(func)


def vjp(func: Callable, *primals):
    def grad_fn(*x_in):
        return ivy.to_native(
            func(*ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    primals_out, _vjpfun = ivy.outputs_to_ivy_arrays(jax.vjp)(
        grad_fn, *ivy.to_native(primals, nested=True)
    )

    def vjpfun(x_in):
        return ivy.to_ivy(
            _vjpfun(ivy.to_native(x_in, nested=True)), nested=True, include_derived=True
        )

    return (primals_out, vjpfun)


def jvp(func: Callable, primals, tangents):
    def grad_fn(*x_in):
        return ivy.to_native(
            func(*ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    primals_out, tangents_out = ivy.outputs_to_ivy_arrays(jax.jvp)(
        grad_fn,
        ivy.to_native(primals, nested=True),
        ivy.to_native(tangents, nested=True),
    )

    return (primals_out, tangents_out)
