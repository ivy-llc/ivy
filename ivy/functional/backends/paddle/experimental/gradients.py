# global
from typing import Callable
import paddle

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

    primals_out = func(*ivy.to_ivy(primals, nested=True))

    def vjpfun(x_in):
        _, vjp_result = ivy.to_ivy(
            paddle.incubate.autograd.vjp(
                grad_fn,
                ivy.to_native(primals, nested=True),
                ivy.to_native(x_in, nested=True),
            )
        )
        return ivy.to_ivy(vjp_result, nested=True, include_derived=True)

    return (primals_out, vjpfun)


def jvp(func: Callable, primals, tangents):
    raise IvyNotImplementedException(
        "forward-mode autodiff not available for paddle backend"
    )
