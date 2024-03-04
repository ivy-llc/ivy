# global
import ivy
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Callable

# local
from ivy.functional.frontends.torch.nn.parameter import Parameter
from ivy.functional.frontends.torch.tensor import Tensor


class Module(ivy.Module):
    _version: int = 1
    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _modules: Dict[str, Optional["Module"]]

    def __init__(self, *args, device=None, devices=None, **kwargs) -> None:
        super().__init__(
            self,
            *args,
            device=device,
            devices=devices,
            training=True,
            build_mode="explicit",
            dynamic_backend=True,
            **kwargs,
        )
        super().__setattr__("_frontend_module", True)
        super().__setattr__(
            "_attr_mapping", {"_parameters": "v", "_modules": "module_dict"}
        )

    def _create_variables(self, device=None, dtype=None):
        # Create variables stored in the `__dict__` that were set
        # using direct `__setattr__` e.g. self.weight = ...
        v = ivy.Container(
            OrderedDict(
                [
                    (k.replace(".", "/"), v)
                    for k, v in self.__dict__.items()
                    if isinstance(v, Parameter)
                    and not k.startswith(
                        ("_"),
                    )
                ]
            )
        )
        # Created variables that were added using `register_paramter`,
        # since those would appear in `self._v`
        v = (
            ivy.Container(
                OrderedDict(
                    (
                        {
                            _k.replace(".", "/"): _v
                            for (_k, _v) in self._v.items()
                            if _k.replace(".", "/") not in v
                            and not isinstance(_v, ivy.Container)
                        }
                    ),
                    **v,
                )
            )
            if self._v
            else v
        )
        return v

    def _build(self, *args, **kwargs):
        for module in self.__dict__.values():
            if isinstance(module, Module) and module is not self:
                if not module._built:
                    module.build(
                        *module._args,
                        dynamic_backend=module._dynamic_backend,
                        **module._kwargs,
                    )
        return True

    def _replace_update_v(self, new_v, native=None):
        from ivy.functional.ivy.gradients import _is_variable

        native = ivy.default(native, self)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                # noinspection PyProtectedMember
                native.module_dict[k] = self._replace_update_v(v, native.module_dict[k])
            elif isinstance(v, Parameter):
                # noinspection PyProtectedMember
                native.__setattr__(k, v)
            elif _is_variable(v):
                native.__setattr__(k, Parameter(v))
            elif isinstance(v, Tensor):
                # noinspection PyProtectedMember
                native.__setattr__(k, Parameter(v, requires_grad=v.requires_grad))
            else:
                raise ivy.utils.exceptions.IvyException(
                    f"found item in variable container {v} which was neither a sub"
                    " ivy.Container nor a variable."
                )
        return native

    _update_v = _replace_update_v

    def forward(self, *input: Any) -> None:
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )

    def _forward(self, *a, **kw):
        ret = self._call_impl(*a, **kw)
        return ret

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')

        self._modules[name] = module

        super().__setattr__(name, module)

    def apply(self, fn: Callable[["Module"], None]):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def register_buffer(
        self, name: str, value: Optional["Tensor"], persistent: bool = False
    ) -> None:
        super().register_buffer(name, value)

    def register_parameter(self, name: str, value: Optional["Parameter"]) -> None:
        super().register_parameter(name, value)

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise TypeError("`" + item + "` is not an nn.Module")

        return mod

    def get_parameter(self, target: str):
        target = target.replace(".", "/")
        return self.v[target]

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or id(v) in memo or not isinstance(v, Parameter):
                    continue
                if remove_duplicate:
                    memo.add(id(v))
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        gen = self._named_members(
            lambda module: module.v.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        gen = self._named_members(
            lambda module: module.buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        memo = set()
        for name, module in self._module_dict.items():
            if module is not None and id(module) not in memo:
                memo.add(id(module))
                yield name, module

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        if memo is None:
            memo = set()
        if id(self) not in memo:
            if remove_duplicate:
                memo.add(id(self))
            yield prefix, self
            for name, module in self._module_dict.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def _get_name(self):
        return self.__class__.__name__

    def _extra_repr(self) -> str:
        return ""

    def _call_impl(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        if name == "__dict__":
            return super().__getattribute__(name)
        if "_module_dict" in self.__dict__:
            modules = self.__dict__["_module_dict"]
            if name in modules:
                return modules[name]
        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                return buffers[name]
        if "_v" in self.__dict__:
            v = self.__dict__["_v"]
            if name in v:
                return v[name]
        # Adding this attribute mapping s.t if someone tries
        # to retrieve self._modules/self._parameters, we
        # can handle that here
        if "_attr_mapping" in self.__dict__:
            mapping = self.__dict__["_attr_mapping"]
            if name in mapping:
                return super().__getattribute__(mapping[name])
        return super().__getattribute__(name)

    def __setattr__(self, name, value) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_v")
        if params is not None and name in params and isinstance(value, Parameter):
            remove_from(self.__dict__, self._buffers, self._module_dict)
            self.register_parameter(name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self._extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._module_dict.items():
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._v.keys())
        modules = list(self._module_dict.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
