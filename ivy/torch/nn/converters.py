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
        self._all_native_params = dict([(ds, ivy.Container(OrderedDict(
            sorted([(k.replace('.', '/'), v) for k, v in dict(native_module.named_parameters()).items()]))))
            for ds, native_module in zip(self._dev_strs, self._native_modules.values())])

    @staticmethod
    def _inplace_update(p, v):
        p.data = v

    def _inplace_update_v(self, new_v, dev_str):
        ivy.Container.multi_map(lambda xs, kc: self._inplace_update(xs[0], xs[1]),
                                [self._all_native_params[dev_str], new_v])

    def _replace_update_v(self, new_v, dev_str, native=None):
        native = ivy.default(native, self._native_modules[dev_str])
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                # noinspection PyProtectedMember
                native._modules[k] = self._replace_update_v(v, dev_str, native._modules[k])
            elif ivy.is_variable(v):
                # noinspection PyProtectedMember
                native.__setattr__(k, v)
            else:
                raise Exception('found item in variable container {} which was neither a sub ivy.Container'
                                'nor a variable.'.format(v))
        return native

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
