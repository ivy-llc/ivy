# local
import ivy

TO_IGNORE = ["is_variable", "is_ivy_array", "is_native_array", "is_array"]


def _wrap_fn(fn_name):
    def new_fn(
        *args,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
        out=None,
        **kwargs
    ):
        data_idx = ivy.__dict__[fn_name].array_spec[0]
        if (
            not (data_idx[0][0] == 0 and len(data_idx[0]) == 1)
            and args
            and ivy.is_ivy_container(args[0])
        ):
            # if the method has been called as an instance method, and self should not
            # be the first positional arg, then we need to re-arrange and place self
            # in the correct location in the args or kwargs
            self = args[0]
            args = args[1:]
            if len(args) > data_idx[0][0]:
                args = ivy.copy_nest(args, to_mutable=True)
                data_idx = [data_idx[0][0]] + [
                    0 if idx is int else idx for idx in data_idx[1:]
                ]
                ivy.insert_into_nest_at_index(args, data_idx, self)
            else:
                kwargs = ivy.copy_nest(kwargs, to_mutable=True)
                data_idx = [data_idx[0][1]] + [
                    0 if idx is int else idx for idx in data_idx[1:]
                ]
                ivy.insert_into_nest_at_index(kwargs, data_idx, self)

        # return function multi-mapped across the corresponding leaves of the containers
        return ivy.ContainerBase.call_static_multi_map_method(
            fn_name,
            *args,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs
        )

    return new_fn


def add_ivy_container_instance_methods(cls, modules, name_prepend="", to_ignore=()):
    """
    Loop over all ivy modules such as activations, general, etc. and add
    the module functions to ivy container as instance methods using _wrap_fn.
    """
    to_ignore = TO_IGNORE + list(to_ignore)
    for module in modules:
        for key, val in module.__dict__.items():
            full_key = name_prepend + key
            if (
                key.startswith("_")
                or key[0].isupper()
                or not callable(val)
                or full_key in cls.__dict__
                or hasattr(cls, full_key)
                or full_key in to_ignore
                or key not in ivy.__dict__
            ):
                continue
            try:
                setattr(cls, full_key, _wrap_fn(key))
            except AttributeError:
                pass
