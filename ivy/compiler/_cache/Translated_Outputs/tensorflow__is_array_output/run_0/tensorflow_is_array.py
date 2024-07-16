from typing import Any

from .tensorflow__helpers import tensorflow_is_native_array


def tensorflow_is_array(x: Any, /, *, exclusive: bool = False):
    return is_tensorflow_array(x, exclusive=exclusive) or tensorflow_is_native_array(
        x, exclusive=exclusive
    )
