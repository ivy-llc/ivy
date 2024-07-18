from typing import Iterable


def tensorflow__flatten_nest(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from tensorflow__flatten_nest(x)
        else:
            yield x
