"""Collection of Paddle gradient functions, wrapped to fit Ivy syntax and signature."""

# global

from typing import Optional, Callable
import paddle
# local
import ivy
from ivy.utils.exceptions import IvyNotImplementedException


def variable(x, /):
    if ivy.is_int_dtype(x.dtype):
        x = ivy.astype(x, ivy.default_float_dtype()).to_native()
    if not x.is_leaf:
        ret = x.detach()
        ret.stop_gradient = False
        return ret
    ret = x.clone()
    ret.stop_gradient = False
    return ret


def is_variable(x, /, *, exclusive: bool = False):
    raise IvyNotImplementedException()


def variable_data(x: paddle.Tensor, /) -> paddle.Tensor:
    raise IvyNotImplementedException()


def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    raise IvyNotImplementedException()


def value_and_grad(func):
    raise IvyNotImplementedException()


def stop_gradient(
    x: Optional[paddle.Tensor],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def jac(func: Callable):

    raise IvyNotImplementedException()


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        x = ivy.to_native(ivy.array(x_in)).detach()
        x.stop_gradient = False
        grad_fn(x).backward()
        return ivy.to_ivy(x.gradient())

    return callback_fn
