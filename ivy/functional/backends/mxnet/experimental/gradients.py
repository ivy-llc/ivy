# global
from typing import Callable
import mxnet as mx

# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException


def bind_custom_gradient_function(func, custom_grad_fn):
    raise IvyNotImplementedException()


def vjp(func: Callable, *primals):
    def grad_fn(*x_in):
        return ivy.to_native(
            func(*ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    with mx.autograd.record():
        primals_out = grad_fn(*ivy.to_native(primals, nested=True))

    def vjpfun(x_in):
        grads = mx.autograd.grad(
            primals_out,
            ivy.to_native(primals, nested=True),
            head_grads=ivy.to_native(x_in, nested=True),
        )
        return ivy.to_ivy(grads, nested=True, include_derived=True)

    return (ivy.to_ivy(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    raise IvyNotImplementedException()
