# global
from typing import Union, Callable
import numpy as np

# local
import ivy


def reduce(
    operand: np.ndarray,
    init_value: Union[int, float, -float("inf"), float("inf")],
    func: Callable,
    axis: int,
) -> np.ndarray:
    func = ivy.output_to_native_arrays(func)
    return ivy.inputs_to_native_arrays(
        numpy.frompyfunc(func, 2, 1).reduce(operand, axis=axis, initial=init_value)
    )
