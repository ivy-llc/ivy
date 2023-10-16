# global
import logging
from typing import Callable

# local


def bind_custom_gradient_function(func, custom_grad_fn):
    logging.warning(
        "NumPy does not support autograd, 'bind_custom_gradient_function' "
        "has no effect on the array, as gradients are not supported in the first place."
    )
    return func


def vjp(func: Callable, *primals):
    logging.warning(
        "NumPy does not support autograd, 'vjp' returns None in place of `vjpfun`."
    )
    return func(*primals), None


def jvp(func: Callable, primals, tangents):
    logging.warning(
        "NumPy does not support autograd, "
        "'jvp' returns None in place of `tangents_out`."
    )
    return func(*primals), None
