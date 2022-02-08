"""
Collection of general Ivy compilation functions.
"""

# global
from typing import Callable, Any, Union, Tuple, Iterable

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def compile(func: Callable, dynamic: bool = True, example_inputs: Union[Any, Tuple[Any]] = None,
            static_argnums: Union[int, Iterable[int]] = None, static_argnames: Union[int, Iterable[int]] = None,
            f: ivy.Framework = None) -> Callable:
    """
    Provide a function which should be compiled, for faster inference.
    The handle to the newly compiled function is returned.

    :param func: Function to be compiled.
    :type func: callable
    :param dynamic: Whether to compile all conditional branches, regardless of inputs during first invocation.
    :type dynamic: bool, default True
    :param example_inputs: Example of inputs to the function to be compiled.
                            Required for torch in non-dynamic mode, unused by other frameworks.
    :type example_inputs: single input or tuple of inputs.
    :param static_argnums: The argument numbers which should be treated as static for compiling. Default is None.
    :type static_argnums: int or sequence of ints, optional
    :param static_argnames: The argument names which should be treated as static for compiling. Default is None.
    :type static_argnames: str or sequence of strs, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The handle to the newly compiled function.
    """
    return _cur_framework(example_inputs, f=f).compile(
        func, dynamic, example_inputs, static_argnums, static_argnames)
