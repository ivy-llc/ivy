# global
from typing import Callable
import paddle

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)
from ivy.utils.exceptions import IvyNotImplementedException


def bind_custom_gradient_function(func, custom_grad_fn):
    class _CustomModule(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            ret = ivy.to_native(func(x), nested=True, include_derived=True)
            ctx.save_for_backward(x, ret)
            return ret

        @staticmethod
        def backward(ctx, upstream):
            grads = custom_grad_fn(
                *ivy.to_ivy(
                    (ctx.saved_tensor(), upstream), nested=True, include_derived=True
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

    # primals_out = _rebuild_flattened_containers(
    #     grad_fn(*ivy.to_ivy(flattened_primals, nested=True)), ret_idxs
    # )
    primals_out = func(*ivy.to_ivy(primals, nested=True))

    def vjpfun(x_in):
        _, vjp_result = ivy.to_ivy(
            paddle.incubate.autograd.vjp(
                grad_fn,
                ivy.to_native(flattened_primals, nested=True),
                ivy.to_native(_flatten_containers(x_in)[0], nested=True),
            )
        )
        return ivy.to_ivy(
            _rebuild_flattened_containers(vjp_result, ret_idxs),
            nested=True,
            include_derived=True,
        )

    return (ivy.to_ivy(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    raise IvyNotImplementedException()
