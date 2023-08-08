# global
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.utils.backend import current_backend

from ivy.func_wrapper import (
    handle_array_function,
    inputs_to_ivy_arrays,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


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


@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def adagrad_step(
    dcdw: Union[ivy.Array, ivy.NativeArray],
    vt: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    epsilon: float = 0,
    out: Optional[ivy.Array] = None,
) -> Tuple[ivy.Array, ivy.Array]:
    vt = vt + dcdw**2
    return ivy.divide(dcdw, ivy.sqrt(vt) + epsilon, out=out), vt


adagrad_step.out_index = 0


@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def adagrad_update(
    w: Union[ivy.Array, ivy.NativeArray],
    dcdw: Union[ivy.Array, ivy.NativeArray],
    lr: Union[float, ivy.Array, ivy.NativeArray],
    vt_tm1: Union[ivy.Array, ivy.NativeArray],
    step: int,
    /,
    *,
    epsilon: float = 1e-7,
    lr_decay: float = 0,
    stop_gradients: bool = True,
    out: Optional[ivy.Array] = None,
) -> Tuple[ivy.Array, ivy.Array]:
    effective_grads, vt = ivy.adagrad_step(dcdw, vt_tm1, epsilon=epsilon)
    adjusted_lr = lr / (1 + step * lr_decay)
    return (
        ivy.optimizer_update(
            w, effective_grads, adjusted_lr, stop_gradients=stop_gradients, out=out
        ),
        vt,
    )


adagrad_update.out_index = 0
