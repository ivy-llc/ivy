# global
from typing import Callable
import mxnet as mx

# local
import ivy
from ivy.functional.ivy.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)
from ivy.utils.exceptions import IvyNotImplementedException


def bind_custom_gradient_function(func, custom_grad_fn):
    raise IvyNotImplementedException()


def vjp(func: Callable, *primals):
    flattened_primals, ret_idxs = _flatten_containers(primals)

    def grad_fn(*x_in):
        return _flatten_containers(
            ivy.to_native(
                func(
                    *ivy.to_ivy(
                        _rebuild_flattened_containers(x_in, ret_idxs), nested=True
                    )
                ),
                nested=True,
                include_derived=True,
            )
        )

    with mx.autograd.record():
        flat_primals_out, func_ret_idxs = grad_fn(
            *ivy.to_native(flattened_primals, nested=True)
        )

    primals_out = _rebuild_flattened_containers(flat_primals_out, func_ret_idxs)

    def vjpfun(x_in):
        grads = mx.autograd.grad(
            flat_primals_out,
            ivy.to_native(flattened_primals, nested=True),
            head_grads=ivy.to_native(_flatten_containers(x_in)[0], nested=True),
        )

        return _rebuild_flattened_containers(
            ivy.to_ivy(grads, nested=True, include_derived=True), ret_idxs
        )

    return (ivy.to_ivy(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    raise IvyNotImplementedException()
