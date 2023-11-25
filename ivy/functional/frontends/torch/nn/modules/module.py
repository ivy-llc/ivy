from collections import OrderedDict
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
import warnings
import weakref
import ivy
from ivy.functional.frontends.torch.nn.parameter import Parameter

from ivy.functional.frontends.torch.tensor import Tensor
from ivy.functional.frontends.torch.utils import hooks

_grad_t = Union[Tuple[Tensor, ...], Tensor]


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            self.module = weakref.ref(state["module"])


class Module(ivy.Module):
    dump_patches: bool = False
    _version: int = 1
    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_pre_hooks: Dict[int, Callable]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    _forward_hooks_with_kwargs: Dict[int, bool]
    _forward_hooks_always_called: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _modules: Dict[str, Optional["Module"]]
    call_super_init: bool = False
    _compiled_call_impl: Optional[Callable] = None

    def __init__(
        self, *args, device=None, devices=None, inplace_update=False, **kwargs
    ) -> None:
        self.__setattr__("_args", args)
        self.__setattr__("_kwargs", kwargs)
        self.__setattr__(
            "_update_v",
            self._inplace_update_v if inplace_update else self._replace_update_v,
        )
        self.__setattr__("training", True)
        self.__setattr__("_parameters", OrderedDict())
        self.__setattr__("_buffers", OrderedDict())
        self.__setattr__("_non_persistent_buffers_set", set())
        self.__setattr__("_backward_pre_hooks", OrderedDict())
        self.__setattr__("_backward_hooks", OrderedDict())
        self.__setattr__("_is_full_backward_hook", None)
        self.__setattr__("_forward_hooks", OrderedDict())
        self.__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        self.__setattr__("_forward_hooks_always_called", OrderedDict())
        self.__setattr__("_forward_pre_hooks", OrderedDict())
        self.__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        self.__setattr__("_state_dict_hooks", OrderedDict())
        self.__setattr__("_state_dict_pre_hooks", OrderedDict())
        self.__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        self.__setattr__("_load_state_dict_post_hooks", OrderedDict())
        self.__setattr__("_modules", OrderedDict())
        ivy.Module.__init__(
            self, *args, device=device, devices=devices, build_mode="explicit", **kwargs
        )

    def _create_variables(self, device=None, dtype=None):
        return self._native_params

    def _build(self, *args, **kwargs):
        self._native_params = ivy.Container(
            OrderedDict(
                ([
                    (k.replace(".", "/"), v)
                    for k, v in dict(self.named_parameters()).items()
                ])
            ),
            dynamic_backend=False,
        )

    @staticmethod
    def _inplace_update(p, v):
        p.data = v.data

    def _inplace_update_v(self, new_v):
        ivy.Container.cont_multi_map(
            lambda xs, kc: self._inplace_update(xs[0], xs[1]),
            [self._native_params, new_v],
        )

    def _replace_update_v(self, new_v, native=None):
        from ivy.functional.ivy.gradients import _is_variable

        native = ivy.default(native, self)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                # noinspection PyProtectedMember
                native._modules[k] = self._replace_update_v(v, native._modules[k])
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

    def forward(self, *input: Any) -> None:
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "forward" function'
        )

    def _forward(self, *a, **kw):
        ret = self._wrapped_call_impl(*a, **kw)
        return ret

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {type(name)}")
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                f"cannot assign '{type(tensor)}' object to buffer '{name}' "
                "(torch Tensor or None required)"
            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {type(name)}")
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{type(param)}' object to parameter '{name}' "
                "(torch.nn.Parameter or None required)"
            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method."
            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

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
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

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
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def get_parameter(self, target: str):
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: Parameter = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str):
        module_path, _, buffer_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer: Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

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
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
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
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def _get_name(self):
        return self.__class__.__name__

    def _extra_repr(self) -> str:
        return ""

    def register_full_backward_pre_hook(
        self,
        hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> hooks.RemovableHandle:
        handle = hooks.RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_backward_hook(
        self, hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> hooks.RemovableHandle:
        if self._is_full_backward_hook is True:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        self._is_full_backward_hook = False

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_full_backward_hook(
        self,
        hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> hooks.RemovableHandle:
        if self._is_full_backward_hook is False:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        self._is_full_backward_hook = True

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        if prepend:
            self._backward_hooks.move_to_end(handle.id, last=False)
        return handle

    def _get_backward_hooks(self):
        full_backward_hooks: List[Callable] = []
        if _global_is_full_backward_hook is True:
            full_backward_hooks += _global_backward_hooks.values()
        if self._is_full_backward_hook is True:
            full_backward_hooks += self._backward_hooks.values()

        non_full_backward_hooks: List[Callable] = []
        if _global_is_full_backward_hook is False:
            non_full_backward_hooks += _global_backward_hooks.values()
        if self._is_full_backward_hook is False:
            non_full_backward_hooks += self._backward_hooks.values()

        return full_backward_hooks, non_full_backward_hooks

    def _get_backward_pre_hooks(self):
        backward_pre_hooks: List[Callable] = []
        backward_pre_hooks += _global_backward_pre_hooks.values()
        backward_pre_hooks += self._backward_pre_hooks.values()
        return backward_pre_hooks

    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        if not isinstance(result, Tensor):
            if not (
                isinstance(result, tuple) and all(isinstance(r, Tensor) for r in result)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not return a"
                    " single Tensor or a tuple of Tensors is deprecated and will be"
                    " removed in future versions. This hook will be missing some of the"
                    " grad_output. Please use register_full_backward_hook to get the"
                    " documented behavior."
                )
                return
        else:
            result = (result,)

        if not isinstance(inputs, Tensor):
            if not (
                isinstance(inputs, tuple) and all(isinstance(i, Tensor) for i in inputs)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not take as"
                    " input a single Tensor or a tuple of Tensors is deprecated and"
                    " will be removed in future versions. This hook will be missing"
                    " some of the grad_input. Please use register_full_backward_hook to"
                    " get the documented behavior."
                )
                return
        else:
            inputs = (inputs,)

        # At this point we are sure that inputs and result are tuple of Tensors
        out_grad_fn = {r.grad_fn for r in result if r.grad_fn is not None}
        if len(out_grad_fn) == 0 or (
            len(out_grad_fn) == 1 and grad_fn not in out_grad_fn
        ):
            warnings.warn(
                "Using a non-full backward hook when outputs are nested in python data"
                " structure is deprecated and will be removed in future versions. This"
                " hook will be missing some grad_output."
            )
        elif len(out_grad_fn) > 1:
            warnings.warn(
                "Using a non-full backward hook when outputs are generated by different"
                " autograd Nodes is deprecated and will be removed in future versions."
                " This hook will be missing some grad_output. Please use"
                " register_full_backward_hook to get the documented behavior."
            )
        else:
            # At this point the grad_output part of the hook will most likely be correct
            inputs_grad_fn = {i.grad_fn for i in inputs if i.grad_fn is not None}

            next_functions = {n[0] for n in grad_fn.next_functions}

            if inputs_grad_fn != next_functions:
                warnings.warn(
                    "Using a non-full backward hook when the forward contains multiple"
                    " autograd Nodes is deprecated and will be removed in future"
                    " versions. This hook will be missing some grad_input. Please use"
                    " register_full_backward_hook to get the documented behavior."
                )

    def register_forward_pre_hook(
        self, hook, *, prepend: bool = False, with_kwargs: bool = False
    ) -> hooks.RemovableHandle:
        handle = hooks.RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_forward_hook(
        self,
        hook,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> hooks.RemovableHandle:
        handle = hooks.RemovableHandle(
            self._forward_hooks,
            extra_dict=[
                self._forward_hooks_with_kwargs,
                self._forward_hooks_always_called,
            ],
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)
        return handle

    def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)
        else:
            return self._call_impl(*args, **kwargs)

    def _call_impl(self, *args, **kwargs):
        forward_call = self.forward
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (
            self._backward_hooks
            or self._backward_pre_hooks
            or self._forward_hooks
            or self._forward_pre_hooks
            or _global_backward_pre_hooks
            or _global_backward_hooks
            or _global_forward_hooks
            or _global_forward_pre_hooks
        ):
            return forward_call(*args, **kwargs)

        try:
            result = None
            called_always_called_hooks = set()

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = (
                    self._get_backward_hooks()
                )

            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)
                        if args_kwargs_result is not None:
                            if (
                                isinstance(args_kwargs_result, tuple)
                                and len(args_kwargs_result) == 2
                            ):
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple of"
                                    " (new_args, new_kwargs), but got"
                                    f" {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = hooks.BackwardHook(
                    self, full_backward_hooks, backward_pre_hooks
                )
                args = bw_hook.setup_input_hook(args)

            result = forward_call(*args, **kwargs)
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if (
                        hook_id in self._forward_hooks_always_called
                        or hook_id in _global_forward_hooks_always_called
                    ):
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                if not isinstance(result, (Tensor, tuple)):
                    warnings.warn(
                        "For backward hooks to be called,"
                        " module output should be a Tensor or a tuple of Tensors"
                        f" but received {type(result)}"
                    )
                result = bw_hook.setup_output_hook(result)

            # Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, Tensor):
                    if isinstance(var, dict):
                        var = next(v for v in var.values() if isinstance(v, Tensor))
                    else:
                        var = var[0]
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        except Exception:
            # run always called hooks if they have not already been run
            # For now only forward hooks have the always_call option but perhaps
            # this functionality should be added to full backward hooks as well.
            for hook_id, hook in _global_forward_hooks.items():
                if (
                    hook_id in _global_forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):
                    try:
                        hook_result = hook(self, args, result)
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "global module forward hook with ``always_call=True``"
                            " raised an exception that was silenced as another error"
                            f" was raised in forward: {str(e)}"
                        )
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if (
                    hook_id in self._forward_hooks_always_called
                    and hook_id not in called_always_called_hooks
                ):
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)
                        else:
                            hook_result = hook(self, args, result)
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(
                            "module forward hook with ``always_call=True`` raised an"
                            " exception that was silenced as another error was raised"
                            f" in forward: {str(e)}"
                        )
                        continue
            # raise exception raised in try block
            raise

    def __getattribute__(self, name: str) -> Any:
        if name in ("__dict__", "v", "buffers"):
            return super(Module, self).__getattribute__(name)
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        return super(Module, self).__getattribute__(name)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{type(value)}' as child module '{name}' "
                        "(torch.nn.Module or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError(
                            f"cannot assign '{type(value)}' as buffer '{name}' "
                            "(torch.Tensor or None expected)"
                        )
                    for hook in _global_buffer_registration_hooks.values():
                        output = hook(self, name, value)
                        if output is not None:
                            value = output
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _register_state_dict_hook(self, hook):
        handle = hooks.RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook):
        handle = hooks.RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
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
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)


# --- Helpers --- #
# --------------- #


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s

    return s


# --- Main --- #
# ------------ #


def register_module_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]
) -> hooks.RemovableHandle:
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = False

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_buffer_registration_hook(
    hook: Callable[..., None]
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None], *, always_call: bool = False
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def register_module_forward_pre_hook(
    hook: Callable[..., None]
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]
) -> hooks.RemovableHandle:
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    _global_is_full_backward_hook = True

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
    hook: Callable[["Module", _grad_t], Union[None, _grad_t]]
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None]
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None]
) -> hooks.RemovableHandle:
    handle = hooks.RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()
r"""
This tracks hooks common to all modules that are executed before/after calling forward
and backward.

This is global state used for debugging/profiling purposes
"""
_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_backward_hooks: Dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()
