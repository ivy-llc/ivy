from typing import Callable
from ..numpy.func_wrapper import to_ivy_arrays_and_back as to_ivy_arrays_and_back_numpy


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    return to_ivy_arrays_and_back_numpy(fn)
