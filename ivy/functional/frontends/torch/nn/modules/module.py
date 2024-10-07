# global
import ivy
from collections import OrderedDict
import typing
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Callable
import threading
import itertools
import warnings 

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
            if hasattr(module, "apply"):
                module.apply(fn)
            else:
                fn(module)
        fn(self)
        return self

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
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
                if v is None or id(v) in memo:
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
                if not hasattr(module, "named_modules"):
                    yield submodule_prefix, self
                else:
                    yield from module.named_modules(
                        memo, submodule_prefix, remove_duplicate
                    )

    def _load_from_state_dict(
        self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs
    ):
        def _retrive_layer(model, key):
            if len(key.split(".")) == 1:
                return model, key

            module_path, weight_name = key.rsplit(".", 1)

            # Retrieve the layer using the module path
            layer = model
            for attr in module_path.split("."):
                layer = getattr(layer, attr)

            return layer, weight_name

        persistent_buffers = {k: v for k, v in self._buffers.items()}
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not ivy.is_array(input_param):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected ArrayLike object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                if not isinstance(input_param, Parameter):
                    input_param = Parameter(input_param)

                layer, weight_name = _retrive_layer(self, name)
                try:
                    setattr(layer, weight_name, input_param)
                except Exception as ex:
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        f"whose dimensions in the model are {param.shape} and "
                        f"whose dimensions in the checkpoint are {input_param.shape}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :].split(".", 1)
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self, state_dict: typing.Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.
        """
        if not isinstance(state_dict, typing.Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        state_dict = ivy.nested_map(
            lambda x: ivy.native_array(x.numpy()),
            state_dict,
            include_derived=True,
            shallow=True,
        )
        state_dict = OrderedDict(state_dict)

        def load(module, local_state_dict, prefix=""):
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            #TODO: maybe we should implement this similar to PT
            # and make this recursive. 

        load(self, state_dict)
        del load

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if strict:
            missing_keys = sorted(missing_keys)
            unexpected_keys = sorted(unexpected_keys)
            if len(missing_keys) > 0:
                warnings.warn(
                    "Missing key(s) in state_dict: {}\n".format(
                        ", ".join(f"'{k}'" for k in missing_keys)
                    )
                )
            if len(unexpected_keys) > 0:
                warnings.warn(
                    "Unexpected key(s) in state_dict: {}\n".format(
                        ", ".join(f"'{k}'" for k in unexpected_keys)
                    )
                )
                

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def to(self, *args, **kwargs):
        return self._apply(lambda t: t.to(*args, **kwargs))

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
        state.pop("_thread_local", None)
        state.pop("_metrics_lock", None)
        return state

    def __setstate__(self, state):
        state["_thread_local"] = threading.local()
        state["_metrics_lock"] = threading.Lock()
        self.__dict__.update(state)
