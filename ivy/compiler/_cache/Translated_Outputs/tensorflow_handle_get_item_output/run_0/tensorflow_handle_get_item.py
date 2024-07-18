import functools

from .tensorflow__helpers import tensorflow___getitem__


def tensorflow_handle_get_item(fn):
    @functools.wraps(fn)
    def wrapper(inp, query, **kwargs):
        try:
            res = tensorflow___getitem__(inp, query)
        except IndexError:
            raise
        except Exception:
            res = fn(inp, query, **kwargs)
        return res

    return wrapper
