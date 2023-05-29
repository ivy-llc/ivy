# global
from typing import Union, Callable

import jax

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def reduce(
    operand: JaxArray,
    init_value: Union[int, float, -float("inf"), float("inf")],
    func: Callable,
    axis: int,
) -> JaxArray:
    func = ivy.output_to_native_arrays(func)
    return ivy.inputs_to_native_arrays(
        jax.lax.reduce(operand, init_value, func, (axis,))
    )
