# local
import ivy


def _wrap_fn(fn_name):
    def new_fn(self, *args, **kwargs):
        """
        Add the data of the current array from which the instance function is invoked 
        as the first arg parameter or kwarg parameter. Return the new function with 
        the name fn_name and the new args variable or kwargs as the new inputs.
        """
        fn = ivy.__dict__[fn_name]
        data_idx = fn.array_spec[0]
        if len(args) > data_idx[0][0]:
            args = ivy.copy_nest(args, to_mutable=True)
            data_idx = [data_idx[0][0]] + [
                0 if idx is int else idx for idx in data_idx[1:]
            ]
            ivy.insert_into_nest_at_index(args, data_idx, self._data)
        else:
            kwargs = ivy.copy_nest(kwargs, to_mutable=True)
            data_idx = [data_idx[0][1]] + [
                0 if idx is int else idx for idx in data_idx[1:]
            ]
            ivy.insert_into_nest_at_index(kwargs, data_idx, self._data)
        return fn(*args, **kwargs)

    return new_fn


def add_ivy_array_instance_methods(cls, modules, to_ignore=()):
    """
    Loop over all ivy modules such as activations, general, etc. and add
    the module functions to ivy arrays as instance methods using _wrap_fn.
    """
    for module in modules:
        for key, val in module.__dict__.items():
            if (
                key.startswith("_")
                or key[0].isupper()
                or not callable(val)
                or key in cls.__dict__
                or hasattr(cls, key)
                or key in to_ignore
                or key not in ivy.__dict__
            ):
                continue
            try:
                setattr(cls, key, _wrap_fn(key))
            except AttributeError:
                pass
