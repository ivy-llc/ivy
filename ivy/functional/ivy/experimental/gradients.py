# local
from ivy.utils.backend import current_backend


def bind_custom_gradient_function(func, custom_grad_func):
    """Bind a custom gradient function to a function.

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


def vjp(func, *primals):
    """Compute a (reverse-mode) vector-Jacobian product of `func`.

    Parameters
    ----------
    func : callable
        Function to be differentiated.
    primals
        sequence of primal values at which the Jacobian of `func` should be evaluated.

    Returns
    -------
    ret
        The output of `func` evaluated at `primals`. And a function from a cotangent
        vector representing the vector-Jacobian product of fun evaluated at primals.
    """
    return current_backend(None).vjp(func, *primals)


def jvp(func, primals, tangents):
    """Compute a (forward-mode) Jacobian-vector product of `func`.

    Parameters
    ----------
    func : callable
        Function to be differentiated.
    primals
        sequence of primal values at which the Jacobian of `func` should be evaluated.
    tangents
        sequence of tangent vectors giving the Jacobian-vector product of `func`
        evaluated at `primals`.

    Returns
    -------
    ret
        The output of `func` evaluated at `primals`. And the Jacobian-vector product of
        function evaluated at primals with tangents.
    """
    return current_backend(None).jvp(func, primals, tangents)
