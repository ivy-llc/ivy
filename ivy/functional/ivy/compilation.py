"""Collection of general Ivy compilation functions."""

# global
from typing import Callable, Any, Union, Tuple, Iterable

# local
from ivy.backend_handler import current_backend as _cur_backend


# Extra #
# ------#

# noinspection PyShadowingBuiltins
def compile(
    func: Callable,
    dynamic: bool = True,
    example_inputs: Union[Any, Tuple[Any]] = None,
    static_argnums: Union[int, Iterable[int]] = None,
    static_argnames: Union[int, Iterable[int]] = None,
) -> Callable:
    """Provide a function which should be compiled, for faster inference. The handle to
    the newly compiled function is returned.

    Parameters
    ----------
    func
        Function to be compiled.
    dynamic
        Whether to compile all conditional branches, regardless of inputs during first
        invocation.
    example_inputs
        Example of inputs to the function to be compiled.
        Required for torch in non-dynamic mode, unused by other frameworks.
    static_argnums
        The argument numbers which should be treated as static for compiling. Default is
        None.
    static_argnames
        The argument names which should be treated as static for compiling. Default is
        None.

    Returns
    -------
        The handle to the newly compiled function.
    """
    return _cur_backend(example_inputs).compile(
        func, dynamic, example_inputs, static_argnums, static_argnames
    )
