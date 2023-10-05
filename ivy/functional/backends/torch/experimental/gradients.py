# global
import torch
from typing import Callable

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays


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
    values = []
    ret_idxs = []
    for idx, primal in enumerate(primals):
        if isinstance(primal, ivy.Container):
            grad_arr_idxs = ivy.nested_argwhere(primal, lambda x: ivy.is_array(x))
            grad_arr_values = ivy.multi_index_nest(primal, grad_arr_idxs)
            values.append(grad_arr_values)
            ret_idxs.append(grad_arr_idxs)
        elif ivy.is_array(primal):
            values.append(primal)
            ret_idxs.append(None)

    def grad_fn(*x_in):
        x_in_rebuilt = []
        for idx, ret_idx in enumerate(ret_idxs):
            if ret_idx is None:
                x_in_rebuilt.append(x_in[idx])
            else:
                tmp = ivy.Container()
                ivy.insert_into_nest_at_indices(tmp, ret_idx, values[idx])
                x_in_rebuilt.append(tmp)
        x_in_ivy = ivy.to_ivy(x_in_rebuilt, nested=True)
        func_out = func(*x_in_ivy)
        native_func_out = ivy.to_native(func_out, nested=True, include_derived=True)
        ret = (
            cont.cont_to_dict() if isinstance(cont, ivy.Container) else cont
            for cont in native_func_out
        )
        return tuple(ret)

    primals_out, _vjpfun = ivy.outputs_to_ivy_arrays(torch.func.vjp)(
        grad_fn, *ivy.to_native(values, nested=True)
    )

    def vjp_fun(x_in):
        x_in = ivy.to_native(x_in, nested=True)
        for idx, x in enumerate(x_in):
            assert isinstance(x, type(primals[idx]))
            if isinstance(x, ivy.Container):
                x_in[idx] = x.cont_to_dict()
        return ivy.to_ivy(_vjpfun(tuple(x_in)), nested=True, include_derived=True)

    primals_out = tuple(
        ivy.Container(cont) if isinstance(cont, dict) else cont for cont in primals_out
    )

    return primals_out, _vjpfun


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
