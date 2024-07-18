from .tensorflow__helpers import tensorflow__broadcast_inputs
from .tensorflow__helpers import tensorflow_any


def tensorflow_check_equal(x1, x2, inverse=False, message="", as_array=True):
    def eq_fn(x1, x2):
        return x1 == x2 if inverse else x1 != x2

    def comp_fn(x1, x2):
        return tensorflow_any(eq_fn(x1, x2))

    if not as_array:

        def iter_comp_fn(x1_, x2_):
            return any(eq_fn(x1, x2) for x1, x2 in zip(x1_, x2_))

        def comp_fn(x1, x2):
            return iter_comp_fn(*tensorflow__broadcast_inputs(x1, x2))

    eq = comp_fn(x1, x2)
    if inverse and eq:
        raise Exception(f"{x1} must not be equal to {x2}" if message == "" else message)
    elif not inverse and eq:
        raise Exception(f"{x1} must be equal to {x2}" if message == "" else message)
