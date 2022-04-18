# local
import ivy


def _wrap_fn(fn_name):
    def new_fn(*args, key_chains=None, to_apply=True, prune_unapplied=False, **kwargs):
        arg_cont_idxs = [[i] for i, a in enumerate(args) if ivy.is_ivy_container(a)]
        kwarg_cont_idxs = [[k] for k, v in kwargs.values() if ivy.is_ivy_container(v)]
        arg_conts = ivy.multi_index_nest(args, arg_cont_idxs)
        num_arg_conts = len(arg_conts)
        kwarg_conts = ivy.multi_index_nest(kwargs, kwarg_cont_idxs)
        conts = arg_conts + kwarg_conts
        if not conts:
            raise Exception('no containers found in arguments')
        cont0 = conts[0]
        fn = cont0.ivy.__dict__[fn_name]

        def map_fn(vals, _):
            arg_vals = vals[:num_arg_conts]
            a = ivy.copy_nest(args, to_mutable=True)
            ivy.set_nest_at_indices(a, arg_cont_idxs, arg_vals)
            kwarg_vals = vals[num_arg_conts:]
            kw = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.set_nest_at_indices(kw, kwarg_cont_idxs, kwarg_vals)
            return fn(*a, **kw)

        return ivy.Container.multi_map(map_fn, conts, key_chains, to_apply, prune_unapplied)

    return new_fn


def add_ivy_container_instance_methods(cls, modules, to_ignore=()):
    for module in modules:
        for key, val in module.__dict__.items():
            if key.startswith('_') or key[0].isupper() or not callable(val) or \
                    key in cls.__dict__ or hasattr(cls, key) or key in to_ignore or key not in ivy.__dict__:
                continue
            try:
                setattr(cls, key, _wrap_fn(key))
            except AttributeError:
                pass
