import ivy.functional.frontends.torch.nn as nn

import typing
import operator
from collections import OrderedDict
from collections import abc as container_abcs
from itertools import chain

from .helpers import Translated__addindent


class Translated_ModuleList(nn.Module):
    _modules: typing.Any

    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not -len(self) <= idx < len(self):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __add__(self, other):
        combined = Translated_ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def __repr__(self):
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"
        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue
            start_end_indices.append([i, i])
            repeated_blocks.append(r)
        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"
            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"
            local_repr = Translated__addindent(local_repr, 2)
            lines.append(local_repr)
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def extend(self, modules):
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an iterable, but got "
                + type(modules).__name__
            )
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self
