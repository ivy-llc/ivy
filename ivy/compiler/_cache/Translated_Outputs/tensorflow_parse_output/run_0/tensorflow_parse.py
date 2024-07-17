import collections
from itertools import repeat


def tensorflow_parse(x):
    n = 2
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
