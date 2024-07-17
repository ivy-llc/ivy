from .tensorflow__helpers import tensorflow_add
from .tensorflow__helpers import tensorflow_handle_methods_1


@tensorflow_handle_methods_1
def tensorflow_add_1(input, other, *, alpha=1, out=None):
    return tensorflow_add(input, other, alpha=alpha, out=out)
