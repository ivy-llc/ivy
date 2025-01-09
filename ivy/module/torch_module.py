import os
import importlib.util

import ivy
from tracer.conversion import _to_native

if importlib.util.find_spec("torch"):
    import torch

    class __TorchModule(torch.nn.Module):
        def __init__(self, ivy_module, lazy=False):
            super().__init__()
            self._ivy_module = ivy_module
            self.lazy = lazy
            self._torch_device = "cpu"
            if not lazy:
                self._assign_variables()
                self._parameters_converted = False

        def _assign_variables(self):
            from tracer.conversion import array_to_new_backend

            # TODO: use local ivy.backends.torch here
            ivy.set_backend("torch")
            # Again assuming backend is torch when running this function
            self._ivy_module._v = self._ivy_module._v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )
            ivy_module_weights_in_torch_tensor = self._ivy_module._v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )

            ivy_module_weights_in_torch_tensor.cont_map(
                lambda x, kc: self.register_parameter(
                    name=kc,
                    param=(
                        torch.nn.Parameter(x)
                        if x.is_floating_point() or x.is_complex()
                        else None
                    ),
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
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda _, kc: self._parameters[kc]
                )
                # disabled this behaviour temporarily as it leads to a slowdown with torch.compile
                # ToDo : figure out the cause of the slowdown and uncomment this
                # self._ivy_module._module_graph._container_mode = False
                # self._ivy_module._module_graph._array_mode = False
                self._parameters_converted = True

            # set correct device, so tensors created in the source code (eg. torch.zeros) will be sent to gpu
            with torch.device(self._torch_device):
                # can only use ivy.Module's __call__ only since it has been traced to be used with torch
                ret = self._ivy_module(*args, **kwargs, v=self._ivy_module._v)
                if self._ivy_module._module_graph._array_mode:
                    # Output however could be in ivy.Array form (when ivy_module has not been traced)
                    # So converting to native tensor again
                    nested = True if isinstance(ret, tuple) else False
                    ret = ivy.to_native(ret, nested=nested)

            return ret

        def to_device(self, device):
            ivy.set_backend("torch")
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

        def to(self, *args, **kwargs):
            # send parameters and module graph to the correct device, as well
            # as the torch module + ensure the default device will be set
            # correctly when the module is called (via self._torch_device)

            device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
            if device is not None:
                self._torch_device = device
            # TODO: implement dtype setting of the parameters here

            if isinstance(device, str):
                if "cuda" in device:
                    self.to_device("gpu:0")
                if "cpu" in device:
                    self.to_device("cpu")
            elif isinstance(device, torch.device):
                if "cuda" in device.type:
                    self.to_device("gpu:0")
                if "cpu" in device.type:
                    self.to_device("cpu")

            return super().to(*args, **kwargs)

        def cuda(self):
            self._torch_device = "cuda"
            self.to_device("gpu:0")
            return super().cuda()

        def train(self, mode=True):
            super().train(mode)
            if mode:
                self._ivy_module._module_graph.train()
            else:
                self._ivy_module._module_graph.eval()
            return self

        def eval(self):
            super().eval()
            self._ivy_module._module_graph.eval()
            return self

else:
    print("Torch not available")
