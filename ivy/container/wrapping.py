# local
import ivy

TO_IGNORE = ["is_variable", "is_ivy_array", "is_native_array", "is_array"]


def _wrap_fn(fn_name):
    def new_fn(
        *args, key_chains=None, to_apply=True, prune_unapplied=False, out=None, **kwargs
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
        arg_cont_idxs = ivy.nested_indices_where(
            args, ivy.is_ivy_container, to_ignore=ivy.Container
        )
        kwarg_cont_idxs = ivy.nested_indices_where(
            kwargs, ivy.is_ivy_container, to_ignore=ivy.Container
        )
        # retrieve all the containers in args and kwargs
        arg_conts = ivy.multi_index_nest(args, arg_cont_idxs)
        num_arg_conts = len(arg_conts)
        kwarg_conts = ivy.multi_index_nest(kwargs, kwarg_cont_idxs)
        # Combine the retrieved containers from args and kwargs into a single list
        conts = arg_conts + kwarg_conts
        if not conts:
            raise Exception("no containers found in arguments")
        cont0 = conts[0]
        # Get the function with the name fn_name, enabling containers to specify 
        # their backends irrespective of global ivy's backend
        fn = cont0.ivy.__dict__[fn_name]

        def map_fn(vals, _):
            arg_vals = vals[:num_arg_conts]
            a = ivy.copy_nest(args, to_mutable=True)
            ivy.set_nest_at_indices(a, arg_cont_idxs, arg_vals)
            kwarg_vals = vals[num_arg_conts:]
            kw = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.set_nest_at_indices(kw, kwarg_cont_idxs, kwarg_vals)
            return fn(*a, **kw)
        # Replace each container in arg and kwarg with the arrays at the leaf
        # levels of that container using map_fn and call fn using those arrays
        # as inputs
        ret = ivy.Container.multi_map(
            map_fn, conts, key_chains, to_apply, prune_unapplied
        )
        if ivy.exists(out):
            out.inplace_update(ret)
            ret = out
        return ret

    return new_fn


def add_ivy_container_instance_methods(cls, modules, to_ignore=()):
    """
    Loop over all ivy modules such as activations, general, etc. and add
    the module functions to ivy container as instance methods using _wrap_fn.
    """
    to_ignore = TO_IGNORE + list(to_ignore)
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
