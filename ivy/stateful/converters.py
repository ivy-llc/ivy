"""Converters from Native Modules to Ivy Modules."""

# global
import functools
from typing import Optional, Dict, List
import re  # noqa
import inspect

# local
import ivy

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
    """Convert an instance of a trainable module from a native framework into a
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
        """Convert a Haiku module instance to an Ivy module instance.

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
        try:
            import haiku as hk
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`haiku` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        try:
            from haiku._src.data_structures import FlatMapping  # noqa
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Unable to import `FlatMapping` from `haiku`. Please check if the "
                "requested attribute exists."
            ) from exc

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
        from ivy.stateful.module import _HaikuIvyModule

        return _HaikuIvyModule(
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
        """Convert a Flax module instance to an Ivy module instance.

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
        try:
            import flax  # noqa
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`flax` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        try:
            import jax
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`jax` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

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
        from ivy.stateful.module import _FlaxIvyModule

        return _FlaxIvyModule(
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
        """Convert a Keras module instance to an Ivy module instance.

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
        from ivy.stateful.module import _KerasIvyModule

        return _KerasIvyModule(
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
        """Convert a Paddle layer instance to an Ivy module instance.

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
        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)
        from ivy.stateful.module import _PaddleIvyModule

        return _PaddleIvyModule(
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
        """Convert a Torch module instance to an Ivy module instance.

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
        try:
            import torch  # noqa
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`torch` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        c_args = ivy.default(constructor_args, [])
        c_kwargs = ivy.default(constructor_kwargs, {})
        i_args = ivy.default(instance_args, [])
        i_kwargs = ivy.default(instance_kwargs, {})

        if inspect.isclass(native_module):
            native_module = native_module(*c_args, **c_kwargs)

        from ivy.stateful.module import _TorchIvyModule

        return _TorchIvyModule(
            *i_args,
            native_module=native_module,
            device=device,
            devices=devices,
            inplace_update=inplace_update,
            **i_kwargs,
        )

    def to_keras_module(self):
        """Convert a `ivy.Module` module instance to a `tf.keras.Model`
        instance.

        Returns
        -------
        ret
            The new trainable `tf.keras.Model` instance.
        """
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "`tensorflow` was not found installed on your system. Please proceed "
                "to install it and restart your interpreter to see the changes."
            ) from exc

        class KerasModel(tf.keras.Model):
            def __init__(self, ivy_module):
                super().__init__()
                self._ivy_module = ivy_module
                self._parameters = {}
                self._assign_variables()
                self._populate_params()
                self._propagate_params()

            def _assign_variables(self):
                ivy.set_backend("tensorflow")

                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: (
                        ivy.to_new_backend(x.ivy_array.data, native=True)
                        if hasattr(x, "_ivy_array")
                        else ivy.to_new_backend(x, native=True)
                    ),
                )
                self._ivy_module.v.cont_map(
                    lambda x, kc: (
                        self.add_weight(
                            name=kc, shape=x.shape, dtype=x.dtype, trainable=True
                        )
                        if x is not None
                        else x
                    )
                )
                model_weights = []
                self._ivy_module.v.cont_map(
                    lambda x, kc: (
                        model_weights.append(ivy.to_numpy(x)) if x is not None else x
                    )
                )
                self.set_weights(model_weights)

                ivy.previous_backend()

            def _populate_params(self):
                self._parameters = {
                    re.sub(r":([0-9]+)$", "", param.name).replace(
                        f"{self.name}/", ""
                    ): param
                    for param in self.variables
                }

            def _propagate_params(self):
                def __update_param(ivy_module, x, kc):
                    if kc not in self._parameters:
                        return x
                    # Update param in the underneath ivy module
                    module = ivy_module
                    keys = re.split("[/.]", kc)
                    for key in keys[:-1]:
                        module = module.__getattribute__(key)
                    if hasattr(module, "_update_v"):
                        module._update_v({keys[-1]: self._parameters[kc]})
                    return getattr(module, keys[-1])

                self._ivy_module.v = self._ivy_module.v.cont_map(
                    functools.partial(__update_param, self._ivy_module),
                    inplace=True,
                )

            def call(self, *args, training=None, **kwargs):
                ret = self._ivy_module(*args, **kwargs)
                ret = ivy.nested_map(
                    lambda x: (
                        x.ivy_array.data
                        if hasattr(x, "_ivy_array")
                        else ivy.to_native(x)
                    ),
                    ret,
                )
                return ret

            def __call__(self, *args, **kwargs):
                if ivy.backend != "tensorflow":
                    ivy.set_backend("tensorflow")
                    args, kwargs = ivy.args_to_new_backend(*args, native=True, **kwargs)
                    ivy.previous_backend()
                else:
                    args, kwargs = ivy.args_to_new_backend(*args, native=True, **kwargs)
                return super().__call__(*args, **kwargs)

            def to_device(self, device):
                self._ivy_module._module_graph.to_device(device)
                model_weights = ivy.nested_map(
                    lambda x: (
                        ivy.to_native(ivy.to_device(x, device))
                        if ivy.is_array(x)
                        else x
                    ),
                    self.weights,
                )
                self.set_weights(model_weights)

        keras_module = KerasModel(self)
        return keras_module
