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
            return self.map(
                lambda x, kc: self._ivy.__dict__[fn_name](x, *args, **kwargs) if self._ivy.is_array(x) else x,
                key_chains, to_apply, prune_unapplied, map_sequences, inplace)
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
