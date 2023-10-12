"""Collection of MXNet gradient functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Sequence, Union
import mxnet as mx

# local
from ivy.utils.exceptions import IvyNotImplementedException


def variable(x, /):
    return x


def is_variable(x, /, *, exclusive=False):
    return isinstance(x, mx.ndarray.NDArray)


def variable_data(x, /):
    raise IvyNotImplementedException()


def execute_with_gradients(
    func,
    xs,
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = [[0]],
    ret_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = [[0]],
):
    raise IvyNotImplementedException()


def value_and_grad(func):
    raise IvyNotImplementedException()


def jac(func):
    raise IvyNotImplementedException()


def grad(func, argnums=0):
    raise IvyNotImplementedException()


def stop_gradient(x, /, *, preserve_type=True, out=None):
    raise IvyNotImplementedException()
