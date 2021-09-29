"""
Converter from PyTorch Modules to Ivy Modules
"""

# global
from collections import OrderedDict

# local
import ivy


def to_ivy_module(native_module=None, native_module_class=None, args=None, kwargs=None):

    args = ivy.default(args, [])
    kwargs = ivy.default(kwargs, {})

    if not ivy.exists(native_module):
        if not ivy.exists(native_module_class):
            raise Exception('native_module_class must be specified if native_module is not given')

    class IvyModule(ivy.Module):

        def __init__(self):
            ivy.Module.__init__(self)

        def _create_variables(self, dev_str):
            return self._native_params

        def _build(self):
            self._native_module = ivy.default(
                lambda: native_module, lambda: native_module_class(*args, **kwargs), with_callable=True)
            self._native_params = OrderedDict(
                sorted([(k.replace('.', '_'), v) for k, v in dict(self._native_module.named_parameters()).items()]))
            self._dev_str = ivy.dev_to_str(next(self._native_module.parameters()).device)

        @staticmethod
        def _inplace_update(p, v):
            p.data = v

        def _forward(self, *a, **kw):
            wrapped_mode = ivy.wrapped_mode()
            if wrapped_mode:
                a, kw = ivy.args_to_native(*a, **kw)
            [self._inplace_update(p, v) for p, v in zip(self._native_params.values(), self.v.values())]
            ret = self._native_module(*a, **kw)
            if wrapped_mode:
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)
            return ret

    return IvyModule()
