# global
import torch
from typing import Callable

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)


def bind_custom_gradient_function(func, custom_grad_fn):
    class _CustomModule(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ret = ivy.to_native(func(x), nested=True, include_derived=True)
            ctx.save_for_backward(x, ret)
            return ret

        @staticmethod
        def backward(ctx, upstream):
            grads = custom_grad_fn(
                *ivy.to_ivy(
                    (ctx.saved_tensors, upstream), nested=True, include_derived=True
                )
            )
            return ivy.to_native(grads, nested=True, include_derived=True)

    custom_module = _CustomModule.apply
    return inputs_to_native_arrays(custom_module)


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
        )[0]

    primals_out, _vjpfun = ivy.outputs_to_ivy_arrays(torch.func.vjp)(
        grad_fn, *ivy.to_native(flattened_primals, nested=True)
    )

    primals_out = _rebuild_flattened_containers(primals_out, ret_idxs)

    def vjpfun(x_in):
        return _rebuild_flattened_containers(
            ivy.to_ivy(
                _vjpfun(ivy.to_native(_flatten_containers(x_in)[0], nested=True)),
                nested=True,
                include_derived=True,
            ),
            ret_idxs,
        )

    return (primals_out, vjpfun)


def jvp(func: Callable, primals, tangents):
    def grad_fn(x_in):
        return ivy.to_native(
            func(ivy.to_ivy(x_in, nested=True)), nested=True, include_derived=True
        )

    primals_out, tangents_out = ivy.outputs_to_ivy_arrays(torch.func.jvp)(
        grad_fn,
        ivy.to_native(primals, nested=True),
        ivy.to_native(tangents, nested=True),
    )

    return (primals_out, tangents_out)
