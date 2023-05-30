# global
from typing import Union, Callable, Sequence
import jax

# local
import ivy
from ivy.functional.backends.jax import JaxArray


def reduce(
    operand: JaxArray,
    init_value: Union[int, float],
    func: Callable,
    axes: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> JaxArray:
    func = ivy.output_to_native_arrays(func)
    axes = (axes,) if isinstance(axes, int) else axes
    result = jax.lax.reduce(operand, init_value, func, axes)
    if keepdims:
        result = ivy.expand_dims(result, axis=axes)
    return result
