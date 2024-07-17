from .tensorflow__helpers import tensorflow_add_2
from .tensorflow__helpers import tensorflow_handle_methods_1


@tensorflow_handle_methods_1
def tensorflow_add(arr, other, *, alpha=1):
    return tensorflow_add_2(arr, other, alpha=alpha)
