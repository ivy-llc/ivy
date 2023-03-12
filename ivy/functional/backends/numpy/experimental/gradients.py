# global
import logging


def bind_custom_gradient_function(func, custom_grad_fn):
    logging.warning(
        "NumPy does not support autograd, 'bind_custom_gradient_function' "
        "has no effect on the array, as gradients are not supported in the first place."
    )
    return func
