# global

import paddle
from typing import Callable
import ivy.functional.backends.paddle.gradients as paddle_gradients
import ivy


def bind_custom_gradient_function(func: Callable, custom_grad_fn: Callable) -> Callable:
    def wrapped_func(*args, **kwargs):
        def variable(x):
            if ivy.is_int_dtype(x.dtype):
                x = x.astype(ivy.default_float_dtype())
            if not x.is_leaf:
                ret = x.detach()
                ret.stop_gradient = False
                return ret
            ret = paddle_gradients.copy_array(x).to_native()
            ret.stop_gradient = False
            return ret

        def is_variable(x, *, exclusive: bool = False):
            return isinstance(x, paddle.Tensor) and not x.stop_gradient

        def variable_data(x: paddle.Tensor) -> paddle.Tensor:
            return x.value()

        def _grad_func(y, xs, retain_grads):
            # Compute gradients using Paddle's gradient function
            grads = paddle.grad(
                outputs=y,
                inputs=xs,
                retain_graph=True,
                create_graph=retain_grads,
                allow_unused=True,
            )
            # Handle cases where grads is None (no gradient)
            grads = [
                grad if grad is not None else paddle.zeros_like(x)
                for x, grad in zip(xs, grads)
            ]
            return grads

        def execute_with_gradients(
            func, xs, *, retain_grads=False, xs_grad_idxs=[[0]], ret_grad_idxs=[[0]]
        ):
            # Convert inputs to variables if needed
            xs = [variable(x) if is_variable(x) else x for x in xs]

            # Forward pass through the original function
            y = func(*xs, **kwargs)

            # Compute gradients using Paddle's gradient function
            grads = _grad_func(y, xs, retain_grads)

            # Return the function output and gradients
            return y, grads

        def value_and_grad(func):
            def callback_fn(xs):
                # Execute the function and compute gradients
                ret, grads = execute_with_gradients(func, xs, retain_grads=True)
                return ret, grads

            return callback_fn

        # Convert inputs to variables if needed
        args = [variable(arg) if is_variable(arg) else arg for arg in args]

        # Forward pass through the original function
        y = func(*args, **kwargs)

        # Compute gradients using the custom gradient function
        grads = custom_grad_fn(y, *args, **kwargs)

        # Return the function output and gradients
        return y, grads

    return wrapped_func
