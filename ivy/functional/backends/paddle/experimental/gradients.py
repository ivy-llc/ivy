# global
from typing import Callable

# local


def bind_custom_gradient_function(func: Callable, custom_grad_fn: Callable) -> Callable:
    """
    Bind a custom gradient function to a given Ivy function.

    Parameters:
        func (callable): The Ivy function to which the
        custom gradient function should be bound.
        custom_grad_fn (callable): The custom gradient
        function to be used for computing gradients.

    Returns:
        callable: A wrapped Ivy function that
        uses the custom gradient function for gradients.

    Example:
        def custom_grad(y, x):
            return my_gradient(y, x)

        custom_func = bind_custom_gradient_function(my_function, custom_grad)
    """

    def wrapped_func(*args, **kwargs):
        # Forward pass through the original function
        y = func(*args, **kwargs)

        # Compute gradients using the custom gradient function
        grads = custom_grad_fn(y, *args, **kwargs)

        # Return the function output and gradients
        return y, grads

    return wrapped_func
