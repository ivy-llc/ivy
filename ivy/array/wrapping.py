# local
import ivy


def _wrap_fn(fn_name):
    def new_fn(self, *args, **kwargs):
        arg_array_idxs = ivy.nested_indices_where(args, ivy.is_ivy_array)
        if arg_array_idxs:
            args = ivy.copy_nest(args, to_mutable=True)
            ivy.set_nest_at_index(args, arg_array_idxs[0], self._data)
        else:
            kwarg_array_idxs = ivy.nested_indices_where(kwargs, ivy.is_ivy_array)
            if not kwarg_array_idxs:
                raise Exception('no array arguments found')
            kwargs = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.set_nest_at_index(kwargs, kwarg_array_idxs[0], self._data)
        return ivy.__dict__[fn_name](*args, **kwargs)
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
