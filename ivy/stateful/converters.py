"""Converters from Native Modules to Ivy Modules."""
# global
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

    def to_torch_module(self, lazy):
        import torch

        class TranspiledTorchModule(torch.nn.Module):
            def __init__(self, ivy_module, lazy=False):
                torch.nn.Module.__init__(self)
                self._ivy_module = ivy_module
                self.lazy = lazy
                if not lazy:
                    self._assign_variables()
                    self._parameters_converted = False

            def _assign_variables(self):
                # TODO: use local ivy.backends.torch here
                ivy.set_backend("torch")
                # Again assuming backend is torch when running this function
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: ivy.to_native(x, cont_inplace=True)
                )

                self._ivy_module.v.cont_map(
                    lambda x, kc: self.register_parameter(
                        name=kc, param=torch.nn.Parameter(x)
                    )
                )
                ivy.previous_backend()

            def forward(self, *args, **kwargs):
                if self.lazy:
                    # Convert to ivy first
                    self._ivy_module._module_graph._initialize(*args, **kwargs)
                    self.lazy = False
                    self._assign_variables()
                    self._parameters_converted = False
                # inputs should be only in native tensors
                if self._ivy_module._module_graph and not self._parameters_converted:
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: self._parameters[kc]
                    )
                    self._parameters_converted = True
                # can only use ivy.Module's __call__ only since it has been compiled to
                # be used with torch
                ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
                # Output however could be in ivy.Array form (when ivy_module has
                # not been compiled)
                # So converting to native tensor again
                return ivy.to_native(ret, nested=True)

            def to_device(self, device):
                self._parameters = ivy.nested_map(
                    self._parameters,
                    lambda x: ivy.to_device(x, device) if ivy.is_array(x) else x,
                    include_derived={dict: True},
                )
                self._ivy_module._module_graph.to_device(device)
                if self._parameters_converted:
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: self._parameters[kc]
                    )

        torch_module = TranspiledTorchModule(self, lazy=lazy)

        # set compilation flags
        torch_module._ivy_module._lazy_compiled = lazy
        torch_module._ivy_module._target = "torch"

        return torch_module

    def to_haiku_module(self, lazy):
        import haiku as hk

        ivy_module = self

        class TranspiledHaikuModule(hk.Module):
            def __init__(self):
                super(TranspiledHaikuModule, self).__init__()
                self._ivy_module = ivy_module
                self._parameters_converted = False
                self.lazy = lazy

            def __call__(self, *args, **kwargs):
                if (
                    self.lazy
                    and hasattr(self._ivy_module._module_graph, "_initialized")
                    and not self._ivy_module._module_graph._initialized
                ):
                    # Convert to ivy first
                    self._ivy_module._module_graph._initialize(*args, **kwargs)
                    self.lazy = False
                    self._ivy_module.v.cont_map(
                        lambda x, kc: hk.get_parameter(
                            name=kc,
                            shape=x.shape,
                            dtype=x.dtype,
                            init=lambda shape, dtype: ivy.to_native(
                                self._ivy_module.v[kc], cont_inplace=True
                            ),
                        )
                    )
                # assuming backend is set to JAX when using the call method
                # We do not want to interfere with already set ivy_module.v
                # right now it is a hacky fix.
                if self._ivy_module._module_graph is None:
                    # this is only called during init
                    self._ivy_module.v.cont_map(
                        lambda x, kc: hk.get_parameter(
                            name=kc,
                            shape=x.shape,
                            dtype=x.dtype,
                            init=lambda shape, dtype: ivy.to_native(
                                self._ivy_module.v[kc], cont_inplace=True
                            ),
                        )
                    )
                elif not self._parameters_converted:
                    # if we are using all parameters, we would eventually have to call
                    # `hk.get_parameter` for every param,
                    # so it's okay to call here, won't result in slowdowns
                    # TODO: see if we can remove `array_to_new_backend` from here.
                    prev_backend = ivy.current_backend_str()
                    ivy.set_backend("jax")
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda x, kc: hk.get_parameter(
                            name=kc,
                            shape=x.shape,
                            dtype=x.dtype,
                            init=lambda shape, dtype: ivy.to_native(
                                self._ivy_module.v[kc], cont_inplace=True
                            ),
                        ),  # this won't be used here tho
                    )

                    if prev_backend:
                        ivy.set_backend(prev_backend)
                    self._parameters_converted = True

                args, kwargs = ivy.args_to_native(*args, **kwargs)
                ret = self._ivy_module(*args, v=self._ivy_module.v, **kwargs)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        # set compilation flags
        ivy_module._lazy_compiled = lazy
        ivy_module._target = "jax"

        return TranspiledHaikuModule

    def to_flax_module(self, lazy):
        import flax

        class TranspiledFlaxModule(flax.linen.Module):
            ivy_module: ivy.Module
            lazy: bool = False

            def setup(self):
                self._ivy_module = self.ivy_module
                self._ivy_module._parameters_converted = False
                self._ivy_module.lazy = self.lazy
                if not lazy:
                    self._assign_variables()

            def _assign_variables(self):
                ivy.set_backend("jax")

                self._ivy_module.v.cont_map(
                    lambda x, kc: self.param(
                        # "vars",
                        kc,
                        lambda _, shape, dtype: ivy.to_native(
                            self._ivy_module.v[kc], cont_inplace=True
                        ),
                        x.shape,
                        x.dtype,
                    )
                )
                ivy.previous_backend()

            @flax.linen.compact
            def __call__(self, *args, **kwargs):
                if self._ivy_module.lazy:
                    # Convert to ivy first
                    self._ivy_module._module_graph._initialize(*args, **kwargs)
                    self._ivy_module.lazy = False
                    self._assign_variables()
                    self._ivy_module._parameters_converted = False

                # inputs should be only in native arrays
                if (
                    self._ivy_module._module_graph
                    and not self._ivy_module._parameters_converted
                ):
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: self.variables["params"][kc]
                    )
                    self._ivy_module._parameters_converted = True
                args, kwargs = ivy.args_to_native(*args, **kwargs)
                ret = self._ivy_module(*args, v=self._ivy_module.v, **kwargs)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

        flax_module = TranspiledFlaxModule(self, lazy=lazy)

        # set compilation flags
        flax_module._lazy_compiled = lazy
        flax_module._target = "jax"

        return flax_module

    def to_keras_module(self, lazy):
        import tensorflow as tf

        class TranspiledKerasModel(tf.keras.Model):
            def __init__(self, ivy_module, lazy):
                super(TranspiledKerasModel, self).__init__()
                self._ivy_module = ivy_module
                self._parameters_converted = False
                self.lazy = lazy
                if not lazy:
                    self._assign_variables()

            def _assign_variables(self):
                # TODO: use local ivy.backends.tensorflow here
                ivy.set_backend("tensorflow")

                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: ivy.to_native(x, cont_inplace=True)
                )

                self._ivy_module.v.cont_map(
                    lambda x, kc: self.add_weight(
                        name=kc, shape=x.shape, dtype=x.dtype, trainable=True
                    )
                )
                model_weights = list()
                self._ivy_module.v.cont_map(
                    lambda x, kc: model_weights.append(ivy.to_numpy(x))
                )
                self.set_weights(model_weights)

                ivy.previous_backend()

            def call(self, *args, **kwargs):
                # set model_weights in self._ivy_module.v, so that the
                # graph uses the trainable weights in the computation;
                if self.lazy:
                    # Convert to ivy first
                    kwargs_ = dict(kwargs)
                    del kwargs_["training"]
                    self._ivy_module._module_graph._initialize(*args, **kwargs_)
                    self.lazy = False
                    self._assign_variables()
                    self._parameters_converted = False
                if self._ivy_module._module_graph and not self._parameters_converted:
                    params = {
                        re.sub(r":([0-9]+)$", "", param.name).replace(
                            f"{self.name}/", ""
                        ): param
                        for param in self.variables
                    }
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: params[kc]
                    )
                    self._parameters_converted = True
                # need to call with the weights passed since compilation was done
                # with it
                ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
                if isinstance(ret, tuple):
                    return ivy.args_to_native(*ret)
                return ivy.to_native(ret)

            def __call__(self, *args, **kwargs):
                # ivy.set_backend("tensorflow")
                # args = nest_array_to_new_backend(args, native=True)
                # kwargs = nest_array_to_new_backend(kwargs, native=True)
                # ivy.previous_backend()

                return super(TranspiledKerasModel, self).__call__(*args, **kwargs)

            def to_device(self, device):
                self._ivy_module._module_graph.to_device(device)
                model_weights = ivy.nested_map(
                    self.weights,
                    lambda x: (
                        ivy.to_native(ivy.to_device(x, device))
                        if ivy.is_array(x)
                        else x
                    ),
                )
                self.set_weights(model_weights)

        keras_module = TranspiledKerasModel(self, lazy=lazy)

        # set compilation flags
        keras_module._ivy_module._lazy_compiled = lazy
        keras_module._ivy_module._target = "tensorflow"

        return keras_module

    def to_paddle_module(self, lazy):
        import paddle

        class TranspiledPaddleModule(paddle.nn.Layer):
            def __init__(self, ivy_module, lazy=False):
                super(TranspiledPaddleModule, self).__init__()
                self._ivy_module = ivy_module
                self.lazy = lazy
                if not lazy:
                    self._assign_variables()
                    self._parameters_converted = False

            def _assign_variables(self):
                # TODO: use local ivy.backends.paddle here
                ivy.set_backend("paddle")

                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: ivy.to_native(x, cont_inplace=True)
                )

                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: self.create_parameter(
                        shape=x.shape,
                        dtype=x.dtype,
                        default_initializer=paddle.nn.initializer.Assign(x),
                    )
                )
                ivy.previous_backend()

            def forward(self, *args, **kwargs):
                if self.lazy:
                    self._ivy_module._module_graph._initialize(*args, **kwargs)
                    self.lazy = False
                    self._assign_variables()
                    self._parameters_converted = False
                # inputs should be only in native tensors
                if self._ivy_module._module_graph and not self._parameters_converted:
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: self._parameters[kc]
                    )
                    self._parameters_converted = True

                ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
                return ivy.to_native(ret, nested=True)

            def to_device(self, device):
                self._parameters = ivy.nested_map(
                    self._parameters,
                    lambda x: ivy.to_device(x, device) if ivy.is_array(x) else x,
                    include_derived={dict: True},
                )
                self._ivy_module._module_graph.to_device(device)
                if self._parameters_converted:
                    self._ivy_module.v = self._ivy_module.v.cont_map(
                        lambda _, kc: self._parameters[kc]
                    )

        paddle_module = TranspiledPaddleModule(self, lazy=lazy)

        # set compilation flags
        paddle_module._ivy_module._lazy_compiled = lazy
        paddle_module._ivy_module._target = "paddle"

        return paddle_module
