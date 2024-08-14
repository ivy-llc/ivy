import ivy
from .func_wrapper import use_keras_backend_framework


@use_keras_backend_framework
def variable(x, /):
    pass


@use_keras_backend_framework
def is_variable(x, /, *, exclusive=False):
    pass


@use_keras_backend_framework
def variable_data(x, /):
    pass


@use_keras_backend_framework
def execute_with_gradients(
    func,
    xs,
    /,
    *,
    retain_grads=False,
    xs_grad_idxs=((0,),),
    ret_grad_idxs=((0,),),
):
    pass


@use_keras_backend_framework
def value_and_grad(func):
    pass


@use_keras_backend_framework
def stop_gradient(x, /, *, preserve_type=True, out=None):
    pass


@use_keras_backend_framework
def jac(func):
    pass


@use_keras_backend_framework
def grad(f, argnums=0):
    pass
