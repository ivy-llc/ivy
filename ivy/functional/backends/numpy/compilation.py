"""Collection of Numpy compilation functions."""

# global
import logging
from typing import Any, Callable, Iterable, Optional, Sequence, Union


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
