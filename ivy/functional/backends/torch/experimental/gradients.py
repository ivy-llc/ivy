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
    unique_keys = list(
        {
            ivy.index_nest(ret_idxs, i)
            for i in ivy.nested_argwhere(ret_idxs, lambda x: isinstance(x, str))
        }
    )

    def grad_fn(*x_in):
        ret, idxs = _flatten_containers(
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

        # replave the idxs with the unique keys
        func_ret_idxs = torch.tensor(
            ivy.nested_map(
                lambda x: (
                    unique_keys.index(x)
                    if isinstance(x, str)
                    else -1 if x is None else x
                ),
                idxs,
            )
        )

        return (ret, func_ret_idxs)

    primals_out, _vjpfun, func_ret_idxs = ivy.outputs_to_ivy_arrays(torch.func.vjp)(
        grad_fn, *ivy.to_native(flattened_primals, nested=True), has_aux=True
    )

    func_ret_idxs = ivy.nested_map(
        lambda x: unique_keys[x] if x >= 0 and x < len(unique_keys) else None,
        func_ret_idxs.tolist(),
    )
    primals_out = _rebuild_flattened_containers(primals_out, func_ret_idxs)

    def vjpfun(*x_in):
        ivy.assertions.check_isinstance(x_in, tuple)
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
    flattened_primals, ret_idxs = _flatten_containers(primals)
    flattened_tangents, _ = _flatten_containers(tangents)
    unique_keys = list(
        {
            ivy.index_nest(ret_idxs, i)
            for i in ivy.nested_argwhere(ret_idxs, lambda x: isinstance(x, str))
        }
    )

    def grad_fn(*x_in):
        ret, idxs = _flatten_containers(
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

        # replave the idxs with the unique keys
        func_ret_idxs = torch.tensor(
            ivy.nested_map(
                lambda x: (
                    unique_keys.index(x)
                    if isinstance(x, str)
                    else -1 if x is None else x
                ),
                idxs,
            )
        )

        return (ret, func_ret_idxs)

    primals_out, tangents_out, func_ret_idxs = ivy.outputs_to_ivy_arrays(
        torch.func.jvp
    )(
        grad_fn,
        ivy.to_native(flattened_primals, nested=True),
        ivy.to_native(flattened_tangents, nested=True),
        has_aux=True,
    )

    func_ret_idxs = ivy.nested_map(
        lambda x: unique_keys[x] if x >= 0 and x < len(unique_keys) else None,
        func_ret_idxs.tolist(),
    )

    primals_out = _rebuild_flattened_containers(primals_out, func_ret_idxs)
    tangents_out = _rebuild_flattened_containers(tangents_out, func_ret_idxs)

    return (primals_out, tangents_out)
