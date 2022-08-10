"""Collection of NumPy gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import logging
import ivy


def variable(x):
    logging.warning(
        "NumPy does not support autograd, declaring a 'variable' "
        "is identical to declaring an 'array' when using numpy backend."
    )
    return x


def is_variable(x, exclusive=False):
    # NumPy does not support autograd, checking if x is a variable
    # does have any meaning for NumPy. Return False.
    return False


def variable_data(x):
    return x


def execute_with_gradients(func, xs, retain_grads=False):
    logging.warning(
        "NumPy does not support autograd, "
        "'execute_with_gradients' returns None in place of function gradients."
    )
    xs = ivy.to_ivy(xs)
    func_ret = func(xs)
    if isinstance(func_ret, tuple):
        y = func_ret[0]
        rest = func_ret[1:]
    else:
        y = func_ret
        rest = tuple()
    return (y, None, *rest)


def stop_gradient(x, preserve_type=True, *, out=None):
    logging.warning(
        "NumPy does not support autograd, 'stop_gradient' "
        "has no effect on the array, as gradients are not supported in the first place."
    )
    return x


def jac(func):
    logging.warning(
        "NumPy does not support autograd, 'jac'"
        "returns None in place of function gradients."
    )
    return None


def grad(func):
    logging.warning(
        "NumPy does not support autograd, 'grad'"
        "returns None in place of function gradients."
    )
    return None
