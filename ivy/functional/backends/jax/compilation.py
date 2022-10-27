"""Collection of Jax compilation functions."""

# global
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import jax


def compile(
    fn: Callable,
    /,
    *,
    dynamic: bool = True,
    example_inputs: Optional[Union[Any, Sequence[Any]]] = None,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    return jax.jit(fn, static_argnums=static_argnums, static_argnames=static_argnames)
