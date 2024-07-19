from typing import Any

from .tensorflow__helpers import tensorflow_is_ivy_array_bknd
from .tensorflow__helpers import tensorflow_is_native_array


def tensorflow_is_array_bknd(x: Any, /, *, exclusive: bool = False):
    return tensorflow_is_ivy_array_bknd(
        x, exclusive=exclusive
    ) or tensorflow_is_native_array(x, exclusive=exclusive)
