# global
import abc

# local
import ivy


# noinspection PyUnresolvedReferences
# noinspection PyMissingConstructor
class ContainerBase(dict, abc.ABC):

    def wrap_fn(self, fn_name):
        def new_fn(*args, key_chains=None, to_apply=True, prune_unapplied=False, map_sequences=False, inplace=False,
                   **kwargs):
            arg_cont_idxs = [[i] for i, a in enumerate(args) if ivy.is_ivy_container(a)]
            kwarg_cont_idxs = [[k] for k, v in kwargs.values() if ivy.is_ivy_container(v)]

            def map_fn(x, kc):
                fn = self._ivy.__dict__[fn_name]
                a_conts = ivy.multi_index_nest(args, arg_cont_idxs)
                kw_conts = ivy.multi_index_nest(kwargs, kwarg_cont_idxs)
                a_vals = [cont[kc] for cont in a_conts]
                kw_vals = [cont[kc] for cont in kw_conts]
                a = ivy.copy_nest(args, to_mutable=True)
                kw = ivy.copy_nest(kwargs, to_mutable=True)
                ivy.set_nest_at_indices(a, arg_cont_idxs, a_vals)
                ivy.set_nest_at_indices(kw, kwarg_cont_idxs, kw_vals)
                return fn(x, *a, **kw) if self._ivy.is_array(x) else x

            return self.map(map_fn, key_chains, to_apply, prune_unapplied, map_sequences, inplace)
        return new_fn

    def __init__(self, module, to_ignore=()):
        for key, val in module.__dict__.items():
            if key.startswith('_') or key[0].isupper() or not callable(val) or \
                    key in self.__dict__ or key in to_ignore or key not in ivy.__dict__:
                continue
            try:
                self.__dict__[key] = self.wrap_fn(key)
            except AttributeError:
                pass
