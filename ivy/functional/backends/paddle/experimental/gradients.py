# global

import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.functional.ivy.gradients import _get_required_float_variables


def bind_custom_gradient_function(func, custom_grad_fn):
    def custom_func(x):
        x, _, _, _, _ = _get_required_float_variables(x, xs_grad_idxs=None)

        # Create a variable from x to enable gradient tracking
        x = x.detach().requires_grad_()

        # Check if we are backpropagating before registering the hook
        if ivy.is_backpropagating():
            ret = func(x)

            def hook_fn(grad):
                custom_grads = custom_grad_fn((x, grad))
                return custom_grads

            x.register_hook(hook_fn)
        else:
            ret = func(x)

        return ret

    return inputs_to_native_arrays(custom_func)
