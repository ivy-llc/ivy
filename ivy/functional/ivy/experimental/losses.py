# global
from typing import Union

# local
import ivy
from ivy.func_wrapper import (
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_array_function,
)
from ivy.utils.exceptions import handle_exceptions


# log_poisson_loss
@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def log_poisson_loss(
    targets: Union[ivy.Array, ivy.NativeArray],
    log_input: Union[ivy.Array, ivy.NativeArray],
    compute_full_loss: bool = False,
    name=None,
) -> ivy.Array:
    try:
        assert targets.shape == log_input.shape
    except ValueError:
        raise ValueError(
            "`log_input` and `targets` must have the same shape, received "
            f"({log_input.shape} vs {targets.shape})."
        )

    result = ivy.exp(log_input) - log_input * targets
    if compute_full_loss:
        point_five = 0.5
        two_pi = 2 * ivy.pi

        stirling_approx = (
            (targets * ivy.log(targets))
            - targets
            + (point_five * ivy.log(two_pi * targets))
        )
        zeros = ivy.zeros_like(targets, dtype=targets.dtype)
        ones = ivy.ones_like(targets, dtype=targets.dtype)
        cond = ivy.logical_and(targets >= zeros, targets <= ones)
        result += ivy.where(cond, zeros, stirling_approx)
    return result
