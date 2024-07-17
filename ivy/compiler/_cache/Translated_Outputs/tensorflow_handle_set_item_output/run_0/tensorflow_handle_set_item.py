import functools

from .tensorflow__helpers import tensorflow___setitem__


def tensorflow_handle_set_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, val, **kwargs):
        try:
            tensorflow___setitem__(inp, query, val)
            res = inp
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, val, **kwargs)
        return res

    return wrapper
