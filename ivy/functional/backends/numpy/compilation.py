"""Collection of Numpy compilation functions."""

# global
from typing import Callable, Any, Union, Sequence, Iterable, Optional
import logging


# noinspection PyUnusedLocal
def compile(
    func: Callable,
    /,
    *,
    dynamic: bool = True,
    example_inputs: Optional[Union[Any, Sequence[Any]]] = None,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    logging.warning(
        "Numpy does not support compiling functions.\n"
        "Now returning the unmodified function."
    )
    return func
