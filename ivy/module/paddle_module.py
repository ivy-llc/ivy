import os
import importlib.util

import ivy
from tracer.conversion import _to_native

if importlib.util.find_spec("paddle"):
    import paddle

    class __PaddleLayer(paddle.nn.Layer):
        def __init__(self, ivy_module, lazy=False):
            super().__init__()
            self._ivy_module = ivy_module
            self.lazy = lazy
            if not lazy:
                self._assign_variables()
                self._parameters_converted = False

        def _create_single_parameter(self, x, kc):
            param = self.create_parameter(
                shape=x.shape,
                dtype=x.dtype,
                default_initializer=None,
            )
            param.set_value(x)
            return param

        def _assign_variables(self):
            from tracer.conversion import array_to_new_backend

            # TODO: use local ivy.backends.paddle here
            ivy.set_backend("paddle")

            self._ivy_module._v = self._ivy_module._v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )
            self._ivy_module._v = self._ivy_module._v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )
            self._ivy_module._v = self._ivy_module._v.cont_map(
                self._create_single_parameter
            )
            for key, item in self._ivy_module._v.cont_to_iterator():
                self.add_parameter(key, item)
            ivy.previous_backend()

        def forward(self, *args, **kwargs):
            if self.lazy:
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self.lazy = False
                self._assign_variables()
                self._parameters_converted = False
            # inputs should be only in native tensors
            if self._ivy_module._module_graph and not self._parameters_converted:
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda _, kc: self._parameters[kc]
                )
                self._parameters_converted = True
                self._ivy_module._module_graph._container_mode = False
                self._ivy_module._module_graph._array_mode = False

            ret = self._ivy_module(*args, **kwargs, v=self._ivy_module._v)
            if self._ivy_module._module_graph._array_mode:
                nested = True if isinstance(ret, tuple) else False
                ret = ivy.to_native(ret, nested=nested)
            return ret

        def to_device(self, device):
            ivy.set_backend("paddle")
            self._parameters = ivy.nested_map(
                lambda x: (
                    ivy.to_native(ivy.to_device(x, device)) if ivy.is_array(x) else x
                ),
                self._parameters,
                include_derived={"dict": True},
            )
            self._ivy_module._module_graph.to_device(device)
            if self._parameters_converted:
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda _, kc: self._parameters[kc]
                )
            ivy.previous_backend()

        def train(self):
            super().train()
            self._ivy_module._module_graph.train()

        def eval(self):
            super().eval()
            self._ivy_module._module_graph.eval()
