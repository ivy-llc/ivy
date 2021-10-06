"""
Converter from PyTorch Modules to Ivy Modules
"""

# global
import copy
from collections import OrderedDict

# local
import ivy


class IvyModule(ivy.Module):

    def __init__(self, native_module_class, native_module, dev_str, dev_strs, inplace_update, *args, **kwargs):
        self._native_module_class = native_module_class
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        self._update_v = self._inplace_update_v if inplace_update else self._replace_update_v
        ivy.Module.__init__(self, dev_str=dev_str, dev_strs=dev_strs)

    def _create_variables(self, dev_str):
        self.vs = dict([(k, ivy.Container(v)) for k, v in self._all_native_params.items()])
        return self._all_native_params[dev_str]

    def _build(self):
        native_module = ivy.default(
            lambda: self._native_module,
            lambda: self._native_module_class(*self._args, **self._kwargs), with_callable=True)
        self._native_modules = dict([(ds, copy.deepcopy(native_module).to(ivy.str_to_dev(ds)))
                                     for ds in self._dev_strs])
        self._all_native_params = dict([(ds, OrderedDict(
            sorted([(k.replace('.', '_'), v) for k, v in dict(native_module.named_parameters()).items()])))
            for ds, native_module in zip(self._dev_strs, self._native_modules.values())])

    @staticmethod
    def _inplace_update(p, v):
        p.data = v

    def _inplace_update_v(self, new_v, dev_str):
        [self._inplace_update(p, v) for p, v in zip(self._all_native_params[dev_str].values(), new_v.values())]

    def _replace_update_v(self, new_v, dev_str):
        for (k, p), v in zip(self._all_native_params[dev_str].items(), new_v.values()):
            key_split = k.split('_')
            native_key = '_'.join(key_split[:-1])
            var_type = key_split[-1]
            # noinspection PyProtectedMember
            self._native_modules[dev_str]._modules[native_key].__setattr__(var_type, v)

    def _forward(self, *a, **kw):
        wrapped_mode = ivy.wrapped_mode()
        if wrapped_mode:
            a, kw = ivy.args_to_native(*a, **kw)
        dev_str = self.v.dev_str
        self._update_v(self.v, dev_str)
        ret = self._native_modules[dev_str](*a, **kw)
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

    return IvyModule(native_module_class, native_module, dev_str, dev_strs, inplace_update, *args, **kwargs)
