"""Collection of Paddle gradient functions, wrapped to fit Ivy syntax and signature."""

# global

from typing import Optional, Callable
import paddle
# local
import ivy
from ivy.exceptions import IvyNotImplementedException


def variable(x, /):
    raise IvyNotImplementedException()


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
    raise IvyNotImplementedException()
