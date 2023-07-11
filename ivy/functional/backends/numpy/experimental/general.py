# global
from typing import Union, Callable, Sequence
import numpy as np

# local
from . import backend_version
from ivy import with_unsupported_dtypes


@with_unsupported_dtypes({"1.25.1 and below": ("complex",)}, backend_version)
def reduce(
    operand: np.ndarray,
    init_value: Union[int, float],
    computation: Callable,
    /,
    *,
    axes: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> np.ndarray:
    axes = (
        (axes,)
        if isinstance(axes, int)
        else tuple(axes) if isinstance(axes, list) else axes
    )
    reduced_func = np.frompyfunc(computation, 2, 1).reduce
    op_dtype = operand.dtype
    for axis in axes:
        operand = reduced_func(operand, axis=axis, initial=init_value, keepdims=True)
    if not keepdims:
        operand = np.squeeze(operand, axis=axes)
    return operand.astype(op_dtype)
