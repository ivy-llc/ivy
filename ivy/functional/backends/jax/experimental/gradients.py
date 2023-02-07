# global
import jax

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
