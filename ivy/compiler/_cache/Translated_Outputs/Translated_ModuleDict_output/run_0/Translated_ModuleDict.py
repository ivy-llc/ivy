import ivy.functional.frontends.torch.nn as nn

import typing
from collections import OrderedDict
from collections import abc as container_abcs


class Translated_ModuleDict(nn.Module):
    _modules: typing.Any

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        self._modules.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def update(self, modules):
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an iterable of key/value pairs, but got "
                + type(modules).__name__
            )
        if isinstance(
            modules, (OrderedDict, Translated_ModuleDict, container_abcs.Mapping)
        ):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element #"
                        + str(j)
                        + " should be Iterable; is"
                        + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element #"
                        + str(j)
                        + " has length "
                        + str(len(m))
                        + "; 2 is required"
                    )
                self[m[0]] = m[1]
