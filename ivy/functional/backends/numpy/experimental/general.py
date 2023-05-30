# global
from typing import Union, Callable, Sequence
import numpy as np

# local
import ivy


def reduce(
    operand: np.ndarray,
    init_value: Union[int, float],
    func: Callable,
    axes: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> np.ndarray:
    func = ivy.output_to_native_arrays(func)
    return np.frompyfunc(func, 2, 1).reduce(
        operand, axis=axes, initial=init_value, keepdims=keepdims
    )
