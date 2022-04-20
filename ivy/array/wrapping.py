# global
from typing import get_type_hints

# local
import ivy


def _wrap_fn(fn_name):
    def new_fn(self, *args, **kwargs):
        return ivy.__dict__[fn_name](self._data, *args, **kwargs)
    return new_fn


def add_ivy_array_instance_methods(cls, modules, to_ignore=()):
    for module in modules:
        for key, val in module.__dict__.items():
            if key.startswith('_') or key[0].isupper() or not callable(val) or \
                    key in cls.__dict__ or key in to_ignore or key not in ivy.__dict__:
                continue
            try:
                setattr(cls, key, _wrap_fn(key))
            except AttributeError:
                pass
