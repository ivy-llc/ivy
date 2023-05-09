"""Collection of MXNet gradient functions, wrapped to fit Ivy syntax and signature."""

# global
from typing import Optional, Sequence, Union


def variable(x, /):
    raise NotImplementedError("mxnet.variable Not Implemented")


def is_variable(x, /, *, exclusive=False):
    raise NotImplementedError("mxnet.is_variable Not Implemented")


def variable_data(x, /):
    raise NotImplementedError("mxnet.variable_data Not Implemented")


def execute_with_gradients(
    func,
    xs,
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = None,
    ret_grad_idxs: Optional[Sequence[Sequence[Union[str, int]]]] = None,
):
    raise NotImplementedError("mxnet.execute_with_gradients Not Implemented")


def value_and_grad(func):
    raise NotImplementedError("mxnet.value_and_grad Not Implemented")


def jac(func):
    raise NotImplementedError("mxnet.jac Not Implemented")


def grad(func, argnums=0):
    raise NotImplementedError("mxnet.grad Not Implemented")


def stop_gradient(x, /, *, preserve_type=True, out=None):
    raise NotImplementedError("mxnet.stop_gradient Not Implemented")
