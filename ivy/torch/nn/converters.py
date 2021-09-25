"""
Converter from PyTorch Modules to Ivy Modules
"""

# global
from collections import OrderedDict

# local
import ivy


def to_ivy_module(native_module):

    class IvyModule(ivy.Module):

        def __init__(self):
            self._native_module = native_module
            self._native_params = OrderedDict(
                sorted([(k.replace('.', '_'), v) for k, v in dict(native_module.named_parameters()).items()]))
            ivy.Module.__init__(
                self, dev_str=ivy.dev_to_str(next(native_module.parameters()).device), v=self._native_params)

        @staticmethod
        def _inplace_update(p, v):
            p.data = v

        def _forward(self, *args, **kwargs):
            wrapped_mode = ivy.wrapped_mode()
            if wrapped_mode:
                args, kwargs = ivy.args_to_native(*args, **kwargs)
            [self._inplace_update(p, v) for p, v in zip(self._native_params.values(), self.v.values())]
            ret = self._native_module(*args, **kwargs)
            if wrapped_mode:
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)
            return ret

    return IvyModule()
