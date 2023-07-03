# local
from ivy.utils.backend import current_backend


def bind_custom_gradient_function(func, custom_grad_func):
    """
    Bind a custom gradient function to a function.

    Parameters
    ----------
    func
        Function for which we compute the gradients of the output with respect to.
    custom_grad_func
        Custom gradient function. Should accept a tuple containing the (output, inputs)
        and the upstream gradients w.r.t previous operations.

    Returns
    -------
    ret
        the function
    """
    return current_backend(None).bind_custom_gradient_function(func, custom_grad_func)
