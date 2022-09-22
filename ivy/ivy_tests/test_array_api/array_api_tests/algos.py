__all__ = ["broadcast_shapes"]


from .typing import Shape


# We use a custom exception to differentiate from potential bugs
class BroadcastError(ValueError):
    pass


def _broadcast_shapes(shape1: Shape, shape2: Shape) -> Shape:
    """Broadcasts `shape1` and `shape2`"""
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise BroadcastError

        i = i - 1

    return tuple(shape)


def broadcast_shapes(*shapes: Shape):
    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result
