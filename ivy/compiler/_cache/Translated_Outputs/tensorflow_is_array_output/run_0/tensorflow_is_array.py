from typing import Any

from .tensorflow__helpers import tensorflow_is_ivy_array
from .tensorflow__helpers import tensorflow_is_native_array


def tensorflow_is_array(x: Any, /, *, exclusive: bool = False):
    return tensorflow_is_ivy_array(
        x, exclusive=exclusive
    ) or tensorflow_is_native_array(x, exclusive=exclusive)
