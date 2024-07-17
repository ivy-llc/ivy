from .tensorflow__helpers import tensorflow_is_variable
from .tensorflow__helpers import tensorflow_nested_map


def tensorflow__is_variable(x, exclusive=False, to_ignore=None):
    x = x
    return tensorflow_nested_map(
        lambda x: tensorflow_is_variable(x, exclusive=exclusive),
        x,
        include_derived=True,
        shallow=False,
        to_ignore=to_ignore,
    )
