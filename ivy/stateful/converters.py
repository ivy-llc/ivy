"""Converters from Native Modules to Ivy Modules"""
# global
from types import SimpleNamespace
from typing import Optional, Dict, List
import re
import inspect
from collections import OrderedDict

try:
    import haiku as hk
    from haiku._src.data_structures import FlatMapping
    import jax
except ImportError:
    hk = SimpleNamespace()
    hk.Module = SimpleNamespace
    hk.transform = SimpleNamespace
    hk.get_parameter = SimpleNamespace
    FlatMapping = SimpleNamespace
    jax = SimpleNamespace()
    jax.random = SimpleNamespace()
    jax.random.PRNGKey = SimpleNamespace

try:
    import torch
except ImportError:
    torch = SimpleNamespace()
    torch.nn = SimpleNamespace()
    torch.nn.Parameter = SimpleNamespace
    torch.nn.Module = SimpleNamespace

try:
    import tensorflow as tf
except ImportError:
    tf = SimpleNamespace()
    tf.keras = SimpleNamespace()
    tf.keras.Model = SimpleNamespace

# local
import ivy
from ivy.functional.ivy.gradients import _is_variable
from ivy.backend_handler import current_backend


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
    Convert an instance of a trainable module from a native framework into a
    trainable ivy.Module instance.

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
    def to_haiku_module(self):
        """
        Converts an ivy Module instance to a Haiku Module instance.

        Parameters
        ----------
        ivy_module
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable hk.Module instance.
        """
        ivy_module = self

        class MyHaikuModel(hk.Module):
            def __init__(self):
                super(MyHaikuModel, self).__init__()
                self._ivy_module = ivy_module

            def __call__(self, *args, **kwargs):
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: hk.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: ivy.to_native(self._ivy_module.v[kc]),
                    )
                )
                a, kw = ivy.args_to_native(*args, **kwargs)
                ret = self._ivy_module._forward(*a, **kw)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        return MyHaikuModel

    def to_keras_module(self):
        """
        Converts an ivy Module instance to a Keras Module instance.

        Parameters
        ----------
        self
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable tf.keras.Module instance.
        """
        return MyTFModule(self)

    def to_torch_module(self):
        """
        Converts an ivy Module instance to a Torch Module instance.

        Parameters
        ----------
        self
            The ivy module instance to convert

        Returns
        -------
        ret
            The new trainable torch.nn.Module instance.
        """
        return MyTorchModule(self)

    @staticmethod
    def from_haiku_module(
        native_module,
        constructor_args: Optional[List] = None,
        constructor_kwargs: Optional[Dict] = None,
        instance_args: Optional[List] = None,
        instance_kwargs: Optional[Dict] = None,
        device=None,
        devices=None,
    ):
        """
        Converts a Haiku module instance to an Ivy module instance.

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
            The new trainable torch module instance.

        """
        RNG = jax.random.PRNGKey(42)

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
            def __init__(self, *args, native_module, device, devices, **kwargs):
                self._native_module = native_module
                self._args = args
                self._kwargs = kwargs
                ivy.Module.__init__(
                    self,
                    *args,
                    build_mode="on_init",
                    device=device,
                    devices=devices,
                    **kwargs,
                )

            def _create_variables(self, device, dtype):
                return self._hk_params

            def _build(self, *args, **kwargs):
                args, kwargs = ivy.args_to_native(*args, **kwargs)
                # noinspection PyUnresolvedReferences
                params_hk = self._native_module.init(RNG, *args, **kwargs)
                params_dict = _hk_flat_map_to_dict(params_hk)
                self._hk_params = ivy.Container(params_dict, dynamic_backend=False)
                param_iterator = self._hk_params.cont_to_iterator()
                _, param0 = next(param_iterator)
                self._dev = ivy.as_ivy_dev(param0.device())

            def _forward(self, *a, **kw):
                a, kw = ivy.args_to_native(*a, **kw)
                params_hk = _dict_to_hk_flat_map(self.v.cont_to_dict())
                ret = self._native_module.apply(params_hk, None, *a, **kw)
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
                raise ivy.exceptions.IvyException(
                    "both instance_args and instance_kwargs cannot be none"
                    " when passing a native class"
                )

            def forward_fn(*a, **kw):
                model = native_module(*c_args, **c_kwargs)
                return model(*i_args, **i_kwargs)

            transformed_module = hk.transform(forward_fn)

        return HaikuIvyModule(
            *i_args,
            native_module=transformed_module,
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
        Converts a Keras module instance to an Ivy module instance.

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
                    dynamic_backend=False
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
                raise ivy.exceptions.IvyException(
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
        Converts a Torch module instance to an Ivy module instance.

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
                    dynamic_backend=False
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
                        if isinstance(v, torch.nn.Parameter):
                            # noinspection PyProtectedMember
                            native.__setattr__(k, v)
                        else:
                            # noinspection PyProtectedMember
                            native.__setattr__(k, torch.nn.Parameter(v.data))
                    else:
                        raise ivy.exceptions.IvyException(
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


class MyTorchModule(torch.nn.Module):
    def __init__(self, ivy_module):
        torch.nn.Module.__init__(self)
        self._ivy_module = ivy_module
        self._assign_variables()

    def _assign_variables(self):
        self._ivy_module.v.cont_map(
            lambda x, kc: self.register_parameter(
                name=kc, param=torch.nn.Parameter(ivy.to_native(x))
            )
        )
        self._ivy_module.v = self._ivy_module.v.cont_map(
            lambda x, kc: self._parameters[kc]
        )

    def forward(self, *args, **kwargs):
        a, kw = ivy.args_to_native(*args, **kwargs)
        ret = self._ivy_module._forward(*a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)
        return ivy.to_native(ret)


class MyTFModule(tf.keras.Model):
    def __init__(self, ivy_module):
        super(MyTFModule, self).__init__()
        self._ivy_module = ivy_module
        self._assign_variables()

    def _assign_variables(self):
        self._ivy_module.v.cont_map(
            lambda x, kc: self.add_weight(
                name=kc, shape=x.shape, dtype=x.dtype, trainable=True
            )
        )
        model_weights = list()
        self._ivy_module.v.cont_map(lambda x, kc: model_weights.append(ivy.to_numpy(x)))
        self.set_weights(model_weights)
        params = {re.sub(":\\d+", "", param.name): param for param in self.variables}
        self._ivy_module.v = self._ivy_module.v.cont_map(lambda x, kc: params[kc])

    def call(self, *args, **kwargs):
        a, kw = ivy.args_to_native(*args, **kwargs)
        ret = self._ivy_module._forward(*a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)

        return ivy.to_native(ret)
