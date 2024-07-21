import tensorflow
from collections import OrderedDict
import threading

import typing
from collections import abc as container_abcs

from .tensorflow__stateful import Model as tensorflow_keras_Model
from .tensorflow__helpers import tensorflow__is_variable_bknd
from .tensorflow__helpers import tensorflow_add_frnt_
from .tensorflow__helpers import tensorflow_default_bknd
from .tensorflow__helpers import tensorflow_split_frnt_


class tensorflow_ModuleDict(tensorflow_keras_Model):
    _modules: typing.Any

    def __init__(self, modules=None):
        self.super___init__(
            modules=modules,
            v=getattr(self, "_v", None),
            buffers=getattr(self, "_buffers", None),
            module_dict=getattr(self, "_module_dict", None),
        )
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
            modules, (OrderedDict, tensorflow_ModuleDict, container_abcs.Mapping)
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

    def super___init__(self, *args, device=None, devices=None, **kwargs):
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

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._v.keys())
        modules = list(self._module_dict.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        ag__result_list_0 = []
        for key in keys:
            if not key[0].isdigit():
                res = key
                ag__result_list_0.append(res)
        keys = ag__result_list_0
        return sorted(keys)

    def __getattribute__(self, name):
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
        if "_attr_mapping" in self.__dict__:
            mapping = self.__dict__["_attr_mapping"]
            if name in mapping:
                return super().__getattribute__(mapping[name])
        return super().__getattribute__(name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        state.pop("_thread_local", None)
        state.pop("_metrics_lock", None)
        return state

    def __repr__(self):
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            with tensorflow.name_scope("tensorflow_ModuleDict/extra_lines"):
                extra_lines = tensorflow_split_frnt_(extra_repr, "\n")
        child_lines = []
        for key, module in self._module_dict.items():
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __setattr__(self, name, value):
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_v")
        if (
            params is not None
            and name in params
            and isinstance(value, tensorflow.Variable)
        ):
            remove_from(self.__dict__, self._buffers, self._module_dict)
            self.register_parameter(name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __setstate__(self, state):
        state["_thread_local"] = threading.local()
        state["_metrics_lock"] = threading.Lock()
        self.__dict__.update(state)

    def _build(self, *args, **kwargs):
        for module in self.__dict__.values():
            if isinstance(module, tensorflow_keras_Model) and module is not self:
                if not module._built:
                    module.build(
                        *module._args,
                        dynamic_backend=module._dynamic_backend,
                        **module._kwargs,
                    )
        return True

    def _call_impl(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def _create_variables(self, device=None, dtype=None):
        with tensorflow.name_scope("tensorflow_ModuleDict/v"):
            v = dict(
                OrderedDict(
                    [
                        (k.replace(".", "/"), v)
                        for k, v in self.__dict__.items()
                        if isinstance(v, tensorflow.Variable) and not k.startswith("_")
                    ]
                )
            )
        v = (
            dict(
                OrderedDict(
                    {
                        _k.replace(".", "/"): _v
                        for _k, _v in self._v.items()
                        if _k.replace(".", "/") not in v and not isinstance(_v, dict)
                    },
                    **v,
                )
            )
            if self._v
            else v
        )
        return v

    def _extra_repr(self):
        return ""

    def _forward(self, *a, **kw):
        ret = self._call_impl(*a, **kw)
        return ret

    def _get_name(self):
        return self.__class__.__name__

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate=True
    ):
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
                    tensorflow_add_frnt_(memo, id(v))
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def _replace_update_v(self, new_v, native=None):
        with tensorflow.name_scope("tensorflow_ModuleDict/native"):
            native = tensorflow_default_bknd(native, self)
        for k, v in new_v.items():
            if isinstance(v, dict):
                native.module_dict[k] = self._replace_update_v(v, native.module_dict[k])
            elif isinstance(v, tensorflow.Variable):
                native.__setattr__(k, v)
            elif tensorflow__is_variable_bknd(v):
                native.__setattr__(k, tensorflow.Variable(v))
            elif isinstance(v, tensorflow.Variable):
                native.__setattr__(k, tensorflow.Variable(v))
            else:
                raise Exception(
                    f"found item in variable container {v} which was neither a sub ivy.Container nor a variable."
                )
        return native

    def _update_v(self, new_v, native=None):
        with tensorflow.name_scope("tensorflow_ModuleDict/native"):
            native = tensorflow_default_bknd(native, self)
        for k, v in new_v.items():
            if isinstance(v, dict):
                native.module_dict[k] = self._replace_update_v(v, native.module_dict[k])
            elif isinstance(v, tensorflow.Variable):
                native.__setattr__(k, v)
            elif tensorflow__is_variable_bknd(v):
                native.__setattr__(k, tensorflow.Variable(v))
            elif isinstance(v, tensorflow.Variable):
                native.__setattr__(k, tensorflow.Variable(v))
            else:
                raise Exception(
                    f"found item in variable container {v} which was neither a sub ivy.Container nor a variable."
                )
        return native

    def add_module(self, name, module):
        if (
            not isinstance(
                module, (tensorflow_keras_Model, tensorflow.keras.layers.Layer)
            )
            and module is not None
        ):
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

    def apply(self, fn):
        for module in self.children():
            if hasattr(module, "apply"):
                module.apply(fn)
            else:
                fn(module)
        fn(self)
        return self

    def children(self):
        for _, module in self.named_children():
            yield module

    def call(self, *input):
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )

    def get_parameter(self, target):
        target = target.replace(".", "/")
        return self.pt_v[target]

    def get_submodule(self, target):
        if target == "":
            return self
        atoms: typing.Any = tensorflow_split_frnt_(target, ".")
        mod: typing.Any = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )
            mod = getattr(mod, item)
            if not isinstance(mod, tensorflow_keras_Model):
                raise TypeError("`" + item + "` is not an nn.Module")
        return mod

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
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

    def named_children(self):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        memo = set()
        for name, module in self._module_dict.items():
            if module is not None and id(module) not in memo:
                tensorflow_add_frnt_(memo, id(module))
                yield name, module

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        if memo is None:
            memo = set()
        if id(self) not in memo:
            if remove_duplicate:
                tensorflow_add_frnt_(memo, id(self))
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

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
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

    def parameters(self, recurse=True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def register_buffer(self, name, value, persistent=False):
        super().register_buffer(name, value)

    def register_module(self, name, module):
        self.add_module(name, module)

    def register_parameter(self, name, value):
        super().register_parameter(name, value)

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self
