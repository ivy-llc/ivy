import importlib.util

import ivy
from tracer.conversion import _to_native
from module.utils import _set_jax_module_mode

if importlib.util.find_spec("haiku"):
    import haiku

    class __HaikuModule(haiku.Module):
        ivy_module: ivy.Module
        lazy: bool = False

        def __init__(self):
            super().__init__()
            self._ivy_module = self.ivy_module
            self._parameters_converted = False
            self.lazy = self.lazy

        def __call__(self, *args, **kwargs):
            from tracer.conversion import array_to_new_backend

            if (
                self.lazy
                and hasattr(self._ivy_module._module_graph, "_initialized")
                and not self._ivy_module._module_graph._initialized
            ):
                # Convert to ivy first
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self.lazy = False
                self._ivy_module._v.cont_map(
                    lambda x, kc: haiku.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module._v[kc], inplace=True),
                            native=True,
                        ),
                    )
                )
            # assuming backend is set to JAX when using the call method
            # We do not want to interfere with already set ivy_module._v
            # right now it is a hacky fix.
            if self._ivy_module._module_graph is None:
                # this is only called during init
                self._ivy_module._v.cont_map(
                    lambda x, kc: haiku.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module._v[kc], inplace=True),
                            native=True,
                        ),
                    )
                )
            elif not self._parameters_converted:
                # if we are using all parameters, we would eventually have to call `hk.get_parameter` for every param,
                # so it's okay to call here, won't result in slowdowns
                # TODO: see if we can remove `array_to_new_backend` from here.
                prev_backend = ivy.current_backend_str()
                ivy.set_backend("jax")
                self._ivy_module._v = self._ivy_module._v.cont_map(
                    lambda x, kc: haiku.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module._v[kc], inplace=True),
                            native=True,
                        ),  # this won't be used here tho
                    )
                )
                if prev_backend:
                    ivy.set_backend(prev_backend)
                self._ivy_module._module_graph._container_mode = False
                self._ivy_module._module_graph._array_mode = False
                self._parameters_converted = True

            _set_jax_module_mode(self._ivy_module, kwargs)
            if self._ivy_module._module_graph._array_mode:
                args, kwargs = ivy.args_to_native(*args, **kwargs)
            ret = self._ivy_module(*args, v=self._ivy_module._v, **kwargs)
            if self._ivy_module._module_graph._array_mode:
                nested = True if isinstance(ret, tuple) else False
                ret = ivy.to_native(ret, nested=nested)
            return ret
