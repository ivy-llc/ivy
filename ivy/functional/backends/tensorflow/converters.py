"""Converter from Tensorflow Modules to Ivy Modules."""

# global
import tensorflow as tf
from collections import OrderedDict
import re
# local
import ivy


class IvyModule(ivy.Module):
    def __init__(
            self,
            native_module_class,
            native_module,
            device,
            devices,
            *args,
            **kwargs
    ):
        self._native_module_class = native_module_class
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs

        ivy.Module.__init__(self, device=device, devices=devices)

    def _create_variables(self, device=None, dtype=None):
        return self._native_params

    def _build(self, *args, **kwargs):
        self._native_params = ivy.Container(
            OrderedDict(
                sorted([(param.name, param) for param in self._native_module.variables])
            )
        )

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        ret = self._native_module(*a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)
        return ivy.to_native(ret)


def to_ivy_module(
        native_module=None,
        native_module_class=None,
        args=None,
        kwargs=None,
        device=None,
        devices=None,
):
    args = ivy.default(args, [])
    kwargs = ivy.default(kwargs, {})

    if not ivy.exists(native_module):
        ivy.assertions.check_exists(
            native_module_class,
            message="native_module_class must be specified if native_module is None",
        )

        native_module = native_module_class(**kwargs)
        input_shape = args[0].shape
        native_module.build((input_shape[-1],))

    return IvyModule(
        native_module=native_module,
        device=device,
        devices=devices,
        *args,
        **kwargs
    )


def to_keras_module(ivy_module, args=None, kwargs=None):
    class TFModule(tf.keras.Model, ivy_module):
        def __init__(self):
            super(TFModule, self).__init__()
            ivy_module.__init__(self, **kwargs)
            self._assign_variables()

        def _assign_variables(self):
            self.v.map(
                lambda x, kc: self.add_weight(
                    name=kc,
                    shape=x.shape,
                    dtype=x.dtype,
                    trainable=True
                )
            )
            model_weights = list()
            self.v.map(
                lambda x, kc: model_weights.append(ivy.to_numpy(x)))
            self.set_weights(model_weights)
            params = {
                re.sub(":\\d+", '', param.name): param for param in self.variables
            }
            self.v = self.v.map(lambda x, kc: params[kc])

        def call(self, *args, **kwargs):
            a, kw = ivy.args_to_native(*args, **kwargs)
            ret = self._forward(*a, **kw)
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)

    return TFModule()
