# global
import torch

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
