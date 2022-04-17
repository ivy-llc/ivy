# global
import abc


# noinspection PyUnresolvedReferences
class ArrayBase(abc.ABC):

    def wrap_fn(self, fn):
        def new_fn(*args, **kwargs):
            return fn(self._data, *args, **kwargs)
        return new_fn

    def __init__(self, module):
        for key, val in module.__dict__.items():
            if key.startswith('_') or key[0].isupper() or not callable(val):
                continue
            setattr(self, key, self.wrap_fn(val))
