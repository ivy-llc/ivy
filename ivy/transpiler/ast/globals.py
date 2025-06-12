from enum import Enum, auto
import textwrap


MODULE_TO_ALIAS = {
    "numpy": "np",
    "tensorflow": "tf",
}

TRANSLATED_OUTPUTS_SUBDIR = [
    "torch_frontend_outputs",
    "ivy_outputs",
    "tensorflow_outputs",
    "jax_outputs",
    "numpy_outputs",
]

FRONTEND_STANDARD_GLOBALS = {}

FRONTEND_STANDARD_GLOBALS_TARGET_TO_MODULE = {
    "torch_promotion_table": "ivy.functional.frontends.torch.__init__",
    "numpy_promotion_table": "ivy.functional.frontends.numpy.__init__",
    "numpy_str_to_type_table": "ivy.functional.frontends.numpy.__init__",
    "numpy_scalar_to_dtype": "ivy.functional.frontends.numpy.__init__",
    "numpy_dtype_to_scalar": "ivy.functional.frontends.numpy.__init__",
    "numpy_casting_rules": "ivy.functional.frontends.numpy.__init__",
}

IVY_STANDARD_GLOBALS = {}

IVY_STANDARD_GLOBALS_TARGET_TO_MODULE = {
    "promotion_table": "ivy.__init__",
    "array_api_promotion_table": "ivy.__init__",
}

TF_DUNDERS_MONKEY_PATCH = textwrap.dedent(
    """
from .ivy.functional.frontends.torch.tensor import tensorflow___add___frnt_, tensorflow___sub___frnt_, tensorflow___mul___frnt_, tensorflow___truediv___frnt_, tensorflow___eq___frnt_, tensorflow___ne___frnt_
import tensorflow 

def _define_dunders(orig_method_name):
    original_method = getattr(tensorflow.Tensor, orig_method_name)
    patched_method = {
        '__add__': tensorflow___add___frnt_,
        '__sub__': tensorflow___sub___frnt_,
        '__mul__': tensorflow___mul___frnt_,
        '__truediv__': tensorflow___truediv___frnt_,
        '__eq__': tensorflow___eq___frnt_,
        '__ne__': tensorflow___ne___frnt_,
    }[orig_method_name]

    if orig_method_name in ['__eq__', '__ne__']:
        def impl(self, rhs):
            try:
                res = original_method(self, rhs)
                if isinstance(rhs, (list, tuple)):
                    return False if orig_method_name == '__eq__' else True
                return res
            except Exception:
                return patched_method(self, rhs)
    else:
        def impl(self, rhs):
            try:
                return original_method(self, rhs)
            except Exception:
                return patched_method(self, rhs)

    setattr(tensorflow.Tensor, orig_method_name, impl)

def _define_properties(orig_property_name):
    def device_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.device
        else:
            return self.device

    import keras 
    if keras.__version__ >= '3.0.0':
        patched_method = {
            'device': device_getter,
        }[orig_property_name]
        setattr(tensorflow.keras.Variable, orig_property_name, property(patched_method))

for orig_method_name in ['__add__', '__sub__', '__mul__', '__truediv__', '__eq__', '__ne__']:
    _define_dunders(orig_method_name)

for property_name in ['device']:
    _define_properties(property_name)
"""
)

JAX_DUNDER_PROPERTY_PATCH = textwrap.dedent(
    """
from .ivy.functional.frontends.torch.tensor import jax___add___frnt_, jax___sub___frnt_, jax___mul___frnt_, jax___truediv___frnt_, jax___eq___frnt_, jax___ne___frnt_
import jax 
import jaxlib 
import flax.nnx as nnx 

def _define_dunders(orig_method_name):
    original_method = getattr(jaxlib._jax.ArrayImpl if jax.__version__ >= '0.6.0' else jaxlib.xla_extension.ArrayImpl, orig_method_name)
    patched_method = {
        '__add__': jax___add___frnt_,
        '__sub__': jax___sub___frnt_,
        '__mul__': jax___mul___frnt_,
        '__truediv__': jax___truediv___frnt_,
        '__eq__': jax___eq___frnt_,
        '__ne__': jax___ne___frnt_,
    }[orig_method_name]

    def impl(self, rhs):
        try:
            return original_method(self, rhs)
        except Exception as e:
            return patched_method(self, rhs)

    setattr(jaxlib._jax.ArrayImpl if jax.__version__ >= '0.6.0' else jaxlib.xla_extension.ArrayImpl, orig_method_name, impl)

def _define_properties(orig_property_name):
    def device_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else list(self.value.devices())[0]
        else:
            return list(self.devices())[0]

    def shape_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.shape
        else:
            return original_property.__get__(self)

    def dtype_getter(
        self,
    ):
        if hasattr(self, "value"):
            return None if self.value is None else self.value.dtype
        else:
            return original_property.__get__(self)

    def custom_getattr(self, name):
        if name in ('shape', 'device', 'dtype', 'ndim', 'size', 'itemsize', 'T'):
            value = getattr(self, 'value')
            if value is not None:
                # Attempt to retrieve the attribute from the wrapped object (`value`)
                return getattr(value, name)
        return object.__getattribute__(self, name)
    original_property = getattr(jaxlib._jax.ArrayImpl if jax.__version__ >= '0.6.0' else jaxlib.xla_extension.ArrayImpl, orig_property_name, None)
    patched_method = {
        'device': device_getter,
        'shape': shape_getter,
        'dtype': dtype_getter,
    }[orig_property_name]
    
    setattr(jaxlib._jax.ArrayImpl if jax.__version__ >= '0.6.0' else jaxlib.xla_extension.ArrayImpl, orig_property_name, property(patched_method))
    setattr(nnx.Variable, orig_property_name, property(patched_method))
    setattr(nnx.Variable, '__getattr__', custom_getattr)

for orig_method_name in ['__add__', '__sub__', '__mul__', '__truediv__', '__eq__', '__ne__']:
    _define_dunders(orig_method_name)

for property_name in ['shape', 'dtype', 'device']:
    _define_properties(property_name)

    """
)

BACKEND_STANDARD_GLOBALS = {
    "tensorflow": [
        "\ntf.experimental.numpy.experimental_enable_numpy_behavior(True)\n",
    ],
    "jax": [],
    "numpy": [],
}

MONKEY_PATCH_GLOBALS = {
    "tensorflow": f"\n{TF_DUNDERS_MONKEY_PATCH}\n",
    "jax": f"\n{JAX_DUNDER_PROPERTY_PATCH}\n",
    "numpy": "\n",
}


class TranslatedContext(Enum):
    VARIABLE = auto()
    DECORATOR = auto()
    BASE = auto()
    CLASS_ATTRIBUTE = auto()
    FUNCTION_ARGS = auto()
    TYPE_SPEC = auto()
