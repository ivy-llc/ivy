"""
Converter from JAX Haiku Modules to Ivy Modules
"""

# global
import haiku as hk
# noinspection PyProtectedMember
from haiku._src.data_structures import FlatMapping

# local
import ivy


def _hk_flat_map_to_dict(hk_flat_map):
    ret_dict = dict()
    for k, v in hk_flat_map.items():
        new_k = k.replace('/', '|')
        if isinstance(v, FlatMapping):
            ret_dict[new_k] = _hk_flat_map_to_dict(v)
        else:
            ret_dict[new_k] = v
    return ret_dict


def _dict_to_hk_flat_map(dict_in):
    ret_flat_map = dict()
    for k, v in dict_in.items():
        new_k = k.replace('|', '/')
        if isinstance(v, dict):
            ret_flat_map[new_k] = _dict_to_hk_flat_map(v)
        else:
            ret_flat_map[new_k] = v
    return FlatMapping(ret_flat_map)


class IvyModule(ivy.Module):

    def __init__(self, native_module, dev_str, dev_strs, *args, **kwargs):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        ivy.Module.__init__(self, build_mode='on_call', dev_str=dev_str, dev_strs=dev_strs)

    def _create_variables(self, dev_str):
        return self._hk_params

    def _build(self, *a, **kw):
        if ivy.wrapped_mode():
            a, kw = ivy.args_to_native(*a, **kw)
        # noinspection PyUnresolvedReferences
        params_hk = self._native_module.init(ivy.random.RNG, *a, **kw)
        params_dict = _hk_flat_map_to_dict(params_hk)
        self._hk_params = ivy.Container(params_dict)
        param_iterator = self._hk_params.to_iterator()
        _, param0 = next(param_iterator)
        self._dev_str = ivy.dev_to_str(param0.device())

    def _forward(self, *a, **kw):
        wrapped_mode = ivy.wrapped_mode()
        if wrapped_mode:
            a, kw = ivy.args_to_native(*a, **kw)
        params_hk = _dict_to_hk_flat_map(self.v.to_dict())
        ret = self._native_module.apply(params_hk, None, *a, **kw)
        if wrapped_mode:
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)
        return ret


def to_ivy_module(native_module=None, native_module_class=None, args=None, kwargs=None, dev_str=None, dev_strs=None,
                  inplace_update=False):

    args = ivy.default(args, [])
    kwargs = ivy.default(kwargs, {})

    if not ivy.exists(native_module):
        if not ivy.exists(native_module_class):
            raise Exception('native_module_class must be specified if native_module is not given')

        def forward_fn(*a, **kw):
            model = native_module_class(*args, **kwargs)
            return model(*a, **kw)

        native_module = hk.transform(forward_fn)

    return IvyModule(native_module, dev_str, dev_strs, *args, **kwargs)
