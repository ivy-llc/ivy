"""Converters from Native Modules to Ivy Modules."""
# global
from typing import Optional, Dict, List
import re  # noqa
import inspect
from collections import OrderedDict
import importlib

# local
import ivy
from ivy.functional.ivy.gradients import _is_variable
from ivy.utils.backend import current_backend


def to_ivy_module(
    native_module=None,
    native_module_class=None,
    args=None,
    kwargs=None,
    device=None,
    devices=None,
    inplace_update=False,
):
    """
    Convert an instance of a trainable module from a native framework into a trainable
    ivy.Module instance.

    Parameters
    ----------
    native_module
        The module in the native framework to convert, required if native_module_class
        is not given.
        Default is ``None``.
    native_module_class
        The class of the native module, required if native_module is not given.
        Default is ``None``.
    args
        Positional arguments to pass to the native module class. Default is ``None``.
    kwargs
        Key-word arguments to pass to the native module class. Default is ``None``.
    device
        The device on which to create module variables. Default is ``None``.
    devices
        The devices on which to create module variables. Default is ``None``.
    inplace_update
        For backends with dedicated variable classes, whether to update these inplace.
        Default is ``False``.

    Returns
    -------
    ret
        The new trainable ivy.Module instance.
    """
    return current_backend().to_ivy_module(
        native_module,
        native_module_class,
        args,
        kwargs,
        device,
        devices,
        inplace_update,
    )


class ModuleConverters:
    # Module Converters #
    @staticmethod
    def from_haiku_module(
        native_module,
        params_hk=None,
        rng_seed=0,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Convert a Haiku module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance).
        params_hk
            Haiku parameters to pass to the constructor of the native module.
            Default is ``None``.
        rng_seed
            Seed used to initialize haiku parameters is initializing from a class.
            Default is ``0``.
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable torch module instance.
        """
        hk_spec = importlib.util.find_spec("hk")
        flat_mapping_spec = importlib.util.find_spec(
            "FlatMapping", "haiku._src.data_structures"
        )
        if not hk_spec:
            import haiku as hk
        else:
            hk = importlib.util.module_from_spec(hk_spec)
        if not flat_mapping_spec:
            from haiku._src.data_structures import FlatMapping
        else:
            FlatMapping = importlib.util.module_from_spec(flat_mapping_spec)

        def _hk_flat_map_to_dict(hk_flat_map):
            ret_dict = dict()
            for k, v in hk_flat_map.items():
                new_k = k.replace("/", "|")
                if isinstance(v, FlatMapping):
                    ret_dict[new_k] = _hk_flat_map_to_dict(v)
                else:
                    ret_dict[new_k] = v
            return ret_dict

        def _dict_to_hk_flat_map(dict_in):
            ret_flat_map = dict()
            for k, v in dict_in.items():
                new_k = k.replace("|", "/")
                if isinstance(v, dict):
                    ret_flat_map[new_k] = _dict_to_hk_flat_map(v)
                else:
                    ret_flat_map[new_k] = v
            return FlatMapping(ret_flat_map)

        class HaikuIvyModule(ivy.Module):
            def __init__(
                self, *args, params_hk, native_module, device, devices, **kwargs
            ):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                ivy.Module.__init__(
                    self,
                    params_hk,
                    *args,
                    build_mode="on_init",
                    device=device,
                    devices=devices,
                    **kwargs,
                )

            def _create_variables(self, device, dtype):
                return self._hk_params

            def _build(self, params_hk, *args, **kwargs):
                args, kwargs = ivy.args_to_native(*args, **kwargs)
                # noinspection PyUnresolvedReferences
                params_dict = _hk_flat_map_to_dict(params_hk)
                self._hk_params = ivy.Container(params_dict, dynamic_backend=False)
                param_iterator = self._hk_params.cont_to_iterator()
                _, param0 = next(param_iterator, ["_", 0])
                if hasattr(param0, "device"):
                    self._dev = ivy.as_ivy_dev(param0.device())
                else:
                    self._dev = ivy.as_ivy_dev("cpu")

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                params_hk = _dict_to_hk_flat_map(self.v.cont_to_dict())
                ret = self._native_module.apply(params_hk, 0, *a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        i_args, i_kwargs = ivy.args_to_native(*i_args, **i_kwargs)
        transformed_module = native_module

        if inspect.isclass(native_module):
            if len(i_args) == 0 and len(i_kwargs) == 0:
                raise ivy.utils.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )

            def forward_fn(*a, **kw):
                model = native_module(*c_args, **c_kwargs)
                return model(*i_args, **i_kwargs)

            transformed_module = hk.transform(forward_fn)
            params_hk = transformed_module.init(rng_seed, *i_args, **i_kwargs)

        return HaikuIvyModule(
            *i_args,
            params_hk=params_hk,
            native_module=transformed_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_flax_module(
        native_module,
        params_fx=None,
        rng_seed=0,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Convert a Flax module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance).
        params_fx
            Flax parameters to pass to the constructor of the native module.
            Default is ``None``.
        rng_seed
            Seed used to initialize flax parameters is initializing from a class.
            Default is ``0``.
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """
        flax_spec = importlib.util.find_spec("flax")
        if not flax_spec:
            import flax
        else:
            flax = importlib.util.module_from_spec(flax_spec)
            flax_spec.loader.exec_module(flax)

        jax_spec = importlib.util.find_spec("jax")
        if not jax_spec:
            import jax
        else:
            jax = importlib.util.module_from_spec(jax_spec)
            jax_spec.loader.exec_module(jax)

        class FlaxIvyModule(ivy.Module):
            def __init__(
                self, *args, params_fx, native_module, device, devices, **kwargs
            ):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                ivy.Module.__init__(
                    self,
                    params_fx,
                    *args,
                    build_mode="on_init",
                    device=device,
                    devices=devices,
                    **kwargs,
                )

            def _create_variables(self, device, dtype):
                return self._fx_params

            def _build(self, params_fx, *args, **kwargs):
                args, kwargs = ivy.args_to_native(*args, **kwargs)
                # noinspection PyUnresolvedReferences
                params_dict = flax.core.unfreeze(params_fx)
                self._fx_params = ivy.Container(params_dict, dynamic_backend=False)
                param_iterator = self._fx_params.cont_to_iterator()
                _, param0 = next(param_iterator, ["_", 0])
                self._dev = ivy.as_ivy_dev(ivy.dev(param0))

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                params_fx = flax.core.freeze(self.v.cont_to_dict())
                ret = self._native_module.apply(params_fx, *a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})
        i_args, i_kwargs = ivy.args_to_native(*i_args, **i_kwargs)

        if isinstance(rng_seed, int):
            rng_seed = jax.random.PRNGKey(rng_seed)

        if inspect.isclass(native_module):
            if len(i_args) == 0 and len(i_kwargs) == 0:
                raise ivy.utils.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )

            native_module = native_module(*c_args, **c_kwargs)
            params_fx = native_module.init(rng_seed, *i_args, **i_kwargs)

        return FlaxIvyModule(
            *i_args,
            params_fx=params_fx,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_keras_module(
        native_module=None,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Convert a Keras module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance).
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """

        class KerasIvyModule(ivy.Module):
            def __init__(self, *args, native_module, device, devices, **kwargs):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs

                ivy.Module.__init__(
                    self, *args, device=device, devices=devices, **kwargs
                )

            def _create_variables(self, device=None, dtype=None):
                return self._native_params

            def _build(self, *args, **kwargs):
                self._native_params = ivy.Container(
                    OrderedDict(
                        sorted(
                            [
                                (param.name, param)
                                for param in self._native_module.variables
                            ]
                        )
                    ),
                    dynamic_backend=False,
                )

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                ret = self._native_module(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            if len(i_args) == 0 and len(i_kwargs) == 0:
                raise ivy.utils.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )
            native_module = native_module(*c_args, **c_kwargs)
            input_shape = i_args[0].shape
            native_module.build((input_shape[-1],))

        return KerasIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_paddle_module(
        native_module=None,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Convert a Paddle layer instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance)
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """

        class PaddleIvyModule(ivy.Module):
            def __init__(self, *args, native_module, device, devices, **kwargs):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs

                ivy.Module.__init__(
                    self, *args, device=device, devices=devices, **kwargs
                )

            def _create_variables(self, device=None, dtype=None):
                return self._native_params

            def _build(self, *args, **kwargs):
                self._native_params = ivy.Container(
                    OrderedDict(
                        sorted(
                            [
                                (k.replace(".", "/"), v)
                                for k, v in dict(
                                    self._native_module.named_parameters()
                                ).items()
                            ]
                        )
                    ),
                    dynamic_backend=False,
                )

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                ret = self._native_module(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)

        return PaddleIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            **i_kwargs,
        )

    @staticmethod
    def from_torch_module(
        native_module=None,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
        inplace_update=False,
    ):
        """
        Convert a Torch module instance to an Ivy module instance.

        Parameters
        ----------
        native_module
            The module in the native framework to convert(class or instance)
        constructor_args
            Positional arguments to pass to the constructor of the native module.
            Default is ``None``.
        constructor_kwargs
            Key-word arguments to pass to the constructor of the native module.
             Default is ``None``.
        instance_args
            Positional arguments to pass to the forward pass of the native module.
            Default is ``None``.
        instance_kwargs
            Key-word arguments to pass to the forward pass of the native module.
             Default is ``None``.
        device
            The device on which to create module variables. Default is ``None``.
        devices
            The devices on which to create module variables. Default is ``None``.
        inplace_update
            For backends with dedicated variable classes, whether to update these
            inplace. Default is ``False``.

        Returns
        -------
        ret
            The new trainable ivy.Module instance.
        """
        torch_spec = importlib.util.find_spec("torch")
        if not torch_spec:
            import torch
        else:
            torch = importlib.util.module_from_spec(torch_spec)

        class TorchIvyModule(ivy.Module):
            def __init__(
                self, *args, native_module, device, devices, inplace_update, **kwargs
            ):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                self._update_v = (
                    self._inplace_update_v if inplace_update else self._replace_update_v
                )
                ivy.Module.__init__(
                    self, *args, device=device, devices=devices, **kwargs
                )

            def _create_variables(self, device=None, dtype=None):
                return self._native_params

            def _build(self, *args, **kwargs):
                self._native_params = ivy.Container(
                    OrderedDict(
                        sorted(
                            [
                                (k.replace(".", "/"), v)
                                for k, v in dict(
                                    self._native_module.named_parameters()
                                ).items()
                            ]
                        )
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
                native = ivy.default(native, self._native_module)
                for k, v in new_v.items():
                    if isinstance(v, ivy.Container):
                        # noinspection PyProtectedMember
                        native._modules[k] = self._replace_update_v(
                            v, native._modules[k]
                        )
                    elif _is_variable(v):
                        # noinspection PyProtectedMember
                        native.__setattr__(k, v)
                    elif isinstance(v, torch.Tensor):
                        # noinspection PyProtectedMember
                        native.__setattr__(k, torch.nn.Parameter(v))
                    else:
                        raise ivy.utils.exceptions.IvyException(
                            "found item in variable container {} which was neither a "
                            "sub ivy.Container nor a variable.".format(v)
                        )
                return native

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                self._update_v(self.v)
                ret = self._native_module(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)

        return TorchIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            inplace_update=inplace_update,
            **i_kwargs,
        )
