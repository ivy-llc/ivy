def tensorflow__broadcast_inputs(x1, x2):
    x1_, x2_ = x1, x2
    iterables = list, tuple, tuple
    if not isinstance(x1_, iterables):
        x1_, x2_ = x2, x1
    if not isinstance(x1_, iterables):
        return [x1], [x2]
    if not isinstance(x2_, iterables):
        x1 = [x1] * len(x2)
    return x1, x2
