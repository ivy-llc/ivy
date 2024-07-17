from typing import Any

from .tensorflow__helpers import tensorflow_exists


def tensorflow_default(
    x: Any,
    /,
    default_val: Any,
    *,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
):
    with_callable = catch_exceptions or with_callable
    if rev:
        x, default_val = default_val, x
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return (
        x
        if tensorflow_exists(x)
        else default_val()
        if default_callable
        else default_val
    )
