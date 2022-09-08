"""Collection of general Ivy compilation functions."""

# global
from typing import Callable, Any, Union, Sequence, Iterable, Optional

# local
from ivy.backend_handler import current_backend


# Extra #
# ------#

# noinspection PyShadowingBuiltins
def compile(
    func: Callable,
    /,
    *,
    dynamic: bool = True,
    example_inputs: Optional[Union[Any, Sequence[Any]]] = None,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
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
    ret
        The handle to the newly compiled function.
    """
    return current_backend(example_inputs).compile(
        func,
        dynamic=dynamic,
        example_inputs=example_inputs,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )
