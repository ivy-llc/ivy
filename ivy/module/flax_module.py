import importlib.util

import ivy
from tracer.conversion import _to_native
from module.utils import _set_jax_module_mode

if importlib.util.find_spec("flax"):
    import flax

    class __FlaxModule(flax.linen.Module):
        ivy_module: ivy.Module
        lazy: bool = False

        def setup(self):
            self._ivy_module = self.ivy_module
            self._ivy_module._parameters_converted = False
            self._ivy_module.lazy = self.lazy
            if not self.lazy:
                self._assign_variables()

        def _assign_variables(self):
            from tracer.conversion import array_to_new_backend

            ivy.set_backend("jax")

            self._ivy_module._v.cont_map(
                lambda x, kc: self.param(
                    # "vars",
                    kc,
                    lambda _, shape, dtype: array_to_new_backend(
                        _to_native(self._ivy_module._v[kc], inplace=True),
                        native=True,
                    ),
                    x.shape,
                    x.dtype,
                )
            )
            ivy.previous_backend()

        @flax.linen.compact
        def __call__(self, *args, **kwargs):
            if self._ivy_module.lazy:
                # Convert to ivy first, lazy graph will be
                # initialized when the ivy module is called
                self._ivy_module.lazy = False
                self._assign_variables()
                self._ivy_module._parameters_converted = False

            # inputs should be only in native arrays
            if (
                self._ivy_module._module_graph
                and not self._ivy_module._parameters_converted
            ):
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda _, kc: self.variables["params"][kc]
                )
                self._ivy_module._module_graph._container_mode = False
                self._ivy_module._module_graph._array_mode = False
                self._ivy_module._parameters_converted = True

            _set_jax_module_mode(self._ivy_module, kwargs)
            if self._ivy_module._module_graph._array_mode:
                args, kwargs = ivy.args_to_native(*args, **kwargs)
            ret = self._ivy_module(*args, v=self._ivy_module._v, **kwargs)
            if self._ivy_module._module_graph._array_mode:
                nested = True if isinstance(ret, tuple) else False
                ret = ivy.to_native(ret, nested=nested)
            return ret
