import ivy
from collections import OrderedDict

import typing
import math
from typing import Optional

from .ivy__helpers import ivy__calculate_fan_in_and_fan_out
from .ivy__helpers import ivy__reverse_repeat_tuple
from .ivy__helpers import ivy_add_frnt_
from .ivy__helpers import ivy_empty_frnt
from .ivy__helpers import ivy_kaiming_uniform_
from .ivy__helpers import ivy_split_frnt_
from .ivy__helpers import ivy_uniform_


class ivy__ConvNd(ivy.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[ivy.Array]}

    def _conv_forward(self, input, weight, bias): ...

    in_channels: typing.Any
    _reversed_padding_repeated_twice: typing.Any
    out_channels: typing.Any
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    dilation: typing.Any
    transposed: typing.Any
    output_padding: typing.Any
    groups: typing.Any
    padding_mode: typing.Any
    weight: typing.Any
    bias: typing.Any

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        self.super___init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
            v=getattr(self, "_v", None),
            buffers=getattr(self, "_buffers", None),
            module_dict=getattr(self, "_module_dict", None),
        )
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = ivy__reverse_repeat_tuple(
                self.padding, 2
            )
        if transposed:
            self.weight = ivy.Array(
                ivy_empty_frnt(
                    (*kernel_size, out_channels // groups, in_channels),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = ivy.Array(
                ivy_empty_frnt(
                    (*kernel_size, in_channels // groups, out_channels),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = ivy.Array(ivy_empty_frnt(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        ivy_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = ivy__calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                ivy_uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def super___init__(self, *args, device=None, devices=None, **kwargs):
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

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._v.keys())
        modules = list(self._module_dict.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        keys = [key for key in keys if not key[0].isdigit()]
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
            extra_lines = ivy_split_frnt_(extra_repr, "\n")
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
        if params is not None and name in params and isinstance(value, ivy.Array):
            remove_from(self.__dict__, self._buffers, self._module_dict)
            self.register_parameter(name, value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def _build(self, *args, **kwargs):
        for module in self.__dict__.values():
            if isinstance(module, ivy.Module) and module is not self:
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
        v = ivy.Container(
            OrderedDict(
                [
                    (k.replace(".", "/"), v)
                    for k, v in self.__dict__.items()
                    if isinstance(v, ivy.Array) and not k.startswith("_")
                ]
            )
        )
        v = (
            ivy.Container(
                OrderedDict(
                    {
                        _k.replace(".", "/"): _v
                        for _k, _v in self._v.items()
                        if _k.replace(".", "/") not in v
                        and not isinstance(_v, ivy.Container)
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
        """Helper method for yielding various names + members of modules."""
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
                    ivy_add_frnt_(memo, id(v))
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def _replace_update_v(self, new_v, native=None):
        from ivy.functional.ivy.gradients import _is_variable

        native = ivy.default(native, self)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                native.module_dict[k] = self._replace_update_v(v, native.module_dict[k])
            elif isinstance(v, ivy.Array):
                native.__setattr__(k, v)
            elif _is_variable(v):
                native.__setattr__(k, ivy.Array(v))
            elif isinstance(v, ivy.Array):
                native.__setattr__(k, ivy.Array(v))
            else:
                raise ivy.utils.exceptions.IvyException(
                    f"found item in variable container {v} which was neither a sub ivy.Container nor a variable."
                )
        return native

    def _update_v(self, new_v, native=None):
        from ivy.functional.ivy.gradients import _is_variable

        native = ivy.default(native, self)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                native.module_dict[k] = self._replace_update_v(v, native.module_dict[k])
            elif isinstance(v, ivy.Array):
                native.__setattr__(k, v)
            elif _is_variable(v):
                native.__setattr__(k, ivy.Array(v))
            elif isinstance(v, ivy.Array):
                native.__setattr__(k, ivy.Array(v))
            else:
                raise ivy.utils.exceptions.IvyException(
                    f"found item in variable container {v} which was neither a sub ivy.Container nor a variable."
                )
        return native

    def add_module(self, name, module):
        if not isinstance(module, ivy.Module) and module is not None:
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

    def forward(self, *input):
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )

    def get_parameter(self, target):
        target = target.replace(".", "/")
        return self.v[target]

    def get_submodule(self, target):
        if target == "":
            return self
        atoms: typing.Any = ivy_split_frnt_(target, ".")
        mod: typing.Any = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )
            mod = getattr(mod, item)
            if not isinstance(mod, ivy.Module):
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
                ivy_add_frnt_(memo, id(module))
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
                ivy_add_frnt_(memo, id(self))
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
        """Alias for :func:`add_module`."""
        self.add_module(name, module)

    def register_parameter(self, name, value):
        super().register_parameter(name, value)

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self
