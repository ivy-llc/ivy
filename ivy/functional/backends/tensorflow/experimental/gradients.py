# global
import tensorflow as tf

# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import _get_required_float_variables


def bind_custom_gradient_function(func, custom_grad_fn):
    @tf.custom_gradient
    def custom_module(x):
        x, _, _, _, _ = _get_required_float_variables(x, xs_grad_idxs=None)
        ret = func(x)

        def grad(upstream):
            return custom_grad_fn((x, ret), upstream)

        return ivy.to_native((ret, grad), nested=True, include_derived=True)

    return inputs_to_native_arrays(custom_module)
