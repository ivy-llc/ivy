import os
import re
import importlib.util

import ivy
from tracer.conversion import _to_native

if importlib.util.find_spec("tensorflow"):
    import tensorflow as tf

    class KerasModel(tf.keras.Model):
        def __init__(self, ivy_module, lazy):
            super().__init__()
            self._ivy_module = ivy_module
            self._parameters_converted = False
            self.lazy = lazy
            self._assign_variables()

        def _assign_variables(self):
            from tracer.conversion import array_to_new_backend

            # TODO: use local ivy.backends.tensorflow here
            ivy.set_backend("tensorflow")

            self._ivy_module._v = self._ivy_module._v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )

            self._ivy_module._v = self._ivy_module._v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )
            self._ivy_module._v.cont_map(
                lambda x, kc: self.add_weight(
                    name=kc.replace("/", "_").replace(":", "_"),
                    shape=x.shape,
                    dtype=x.dtype,
                    trainable=True,
                )
            )
            model_weights = []
            self._ivy_module._v.cont_map(
                lambda x, kc: model_weights.append(ivy.to_numpy(x))
            )
            self.set_weights(model_weights)

            ivy.previous_backend()

        def call(self, *args, **kwargs):
            # set model_weights in self._ivy_module._v, so that the
            # graph uses the trainable weights in the computation;
            if self.lazy:
                # Convert to ivy first
                kwargs_ = dict(kwargs)
                if "training" in kwargs_: del kwargs_["training"]
                self._ivy_module._module_graph._initialize(*args, **kwargs_)
                self.lazy = False
                self._parameters_converted = False
            if self._ivy_module._module_graph and not self._parameters_converted:
                params = {
                    re.sub(r":([0-9]+)$", "", param.name).replace(
                        f"{self.name}/", ""
                    ): param
                    for param in self.variables
                }
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda _, kc: params[kc.replace("/", "_").replace(":", "_")]
                ).cont_to_dict()
                self._ivy_module._module_graph._container_mode = False
                self._ivy_module._module_graph._array_mode = False
                self._parameters_converted = True

            # need to call with the weights passed since tracing was done with it
            ret = self._ivy_module(*args, **kwargs, v=self._ivy_module._v)
            if self._ivy_module._module_graph._array_mode:
                nested = isinstance(ret, tuple)
                ret = ivy.to_native(ret, nested=nested)
            return ret

        def __call__(self, *args, **kwargs):
            from tracer.conversion import nest_array_to_new_backend

            if (
                not self._parameters_converted
                or self._ivy_module._module_graph._array_mode
            ):
                ivy.set_backend("tensorflow")
                args = nest_array_to_new_backend(args, native=True)
                kwargs = nest_array_to_new_backend(kwargs, native=True)
                ivy.previous_backend()

            if (
                hasattr(self._ivy_module._module_graph, "_is_trainable_module")
                and self._ivy_module._module_graph._is_trainable_module
                and self._ivy_module._module_graph._traced_train_modes == "all"
            ):
                kwargs_ = dict(kwargs)
                if "training" in kwargs_:
                    if kwargs_["training"]:
                        self._ivy_module._module_graph._train()
                    else:
                        self._ivy_module._module_graph._eval()
                else:
                    # default to eval (keras.backend.learning_phase() has been removed)
                    self._ivy_module._module_graph._eval()

            return super().__call__(*args, **kwargs)

        def to_device(self, device):
            ivy.set_backend("tensorflow")
            self._ivy_module._module_graph.to_device(device)
            model_weights = ivy.nested_map(
                lambda x: (
                    ivy.to_native(ivy.to_device(x, device)) if ivy.is_array(x) else x
                ),
                self.weights,
            )
            self.set_weights(model_weights)
            ivy.previous_backend()
