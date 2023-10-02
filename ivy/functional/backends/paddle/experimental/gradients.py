# global

import paddle
import ivy
from ivy.functional.backends.paddle.gradients import variable
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import _get_required_float_variables


def bind_custom_gradient_function(func, custom_grad_fn):
    def custom_forward(x):
        x, _, _, _, _ = _get_required_float_variables(x, xs_grad_idxs=None)
        ret = func(x)
        return ivy.to_native((ret, x), nested=True, include_derived=True)

    def custom_backward(dy, x):
        # Compute gradients using the custom gradient function
        grads = custom_grad_fn((x, dy))
        return grads

    def custom_func(x):
        ret, x = custom_forward(x)
        # Create a variable from x to ensure gradient tracking
        x = variable(x)
        # Compute gradients using PaddlePaddle's autograd
        x.stop_gradient = False  # Enable gradient tracking
        grads = paddle.autograd.grad(
            outputs=ret,
            inputs=x,
            grad_outputs=None,
            retain_graph=True,  # Retain computation graph for backpropagation
            allow_unused=True,
        )
        # Compute custom gradients
        custom_grads = custom_backward(grads[0], x)
        return ret, custom_grads

    return inputs_to_native_arrays(custom_func)
