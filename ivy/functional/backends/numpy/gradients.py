"""Collection of NumPy gradient functions, wrapped to fit Ivy syntax and
signature."""

# global
import logging
from typing import Sequence, Union
import ivy


def variable(x, /):
    logging.warning(
        "NumPy does not support autograd, declaring a 'variable' "
        "is identical to declaring an 'array' when using numpy backend."
    )
    return x


def is_variable(x, /, *, exclusive=False):
    # NumPy does not support autograd, checking if x is a variable does have any meaning
    # for NumPy. Return False.
    return False


def variable_data(x, /):
    return x


def execute_with_gradients(
    func,
    xs,
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
    ret_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
):
    logging.warning(
        "NumPy does not support autograd, "
        "'execute_with_gradients' returns None in place of function gradients."
    )
    xs = ivy.to_ivy(xs)
    func_ret = func(xs)
    return func_ret, None


def value_and_grad(func):
    logging.warning(
        "NumPy does not support autograd, 'value_and_grad' "
        "has no effect on the array, as gradients are not supported in the first place."
    )

    def grad_fn(xs):
        grads = ivy.nested_map(
            lambda x: ivy.zeros_like(x), xs, include_derived=True, shallow=False
        )
        y = func(xs)
        y = ivy.to_ivy(y)
        return y, grads

    return grad_fn


def jac(func):
    logging.warning(
        "NumPy does not support autograd, 'jac' "
        "has no effect on the array, as gradients are not supported in the first place."
    )

    def grad_fn(xs):
        jacobian = ivy.nested_map(
            lambda x: ivy.zeros_like(x), xs, include_derived=True, shallow=False
        )
        return jacobian

    return grad_fn


def grad(func, argnums=0):
    logging.warning(
        "NumPy does not support autograd, 'grad' "
        "has no effect on the array, as gradients are not supported in the first place."
    )

    def grad_fn(xs):
        grad = ivy.nested_map(
            lambda x: ivy.zeros_like(x), xs, include_derived=True, shallow=False
        )
        y = func(xs)
        y = ivy.to_ivy(y)
        return grad

    return grad_fn


def stop_gradient(x, /, *, preserve_type=True, out=None):
    logging.warning(
        "NumPy does not support autograd, 'stop_gradient' "
        "has no effect on the array, as gradients are not supported in the first place."
    )
    return x
