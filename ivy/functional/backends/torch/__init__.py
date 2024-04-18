# global
import sys
import torch as torch

# local
import ivy
from ivy.func_wrapper import _dtype_from_version

backend_version = {"version": torch.__version__.split("+")[0]}

# Registering ivy.Array as trackable submodule
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.traceable_tensor_subclasses = (ivy.Array,)

# Determine the module in memory based on whether Ivy is local or not
_module_in_memory = (
    sys.modules[__name__] 
    if not ivy.is_local() 
    else sys.modules[ivy.import_module_path].import_cache[__name__]
)

use = ivy.utils.backend.ContextManager(_module_in_memory)

# Native types
NativeArray = torch.Tensor
NativeDevice = torch.device
NativeDtype = torch.dtype
NativeShape = torch.Size

# Sparse array
NativeSparseArray = torch.Tensor

# Devices
valid_devices = ("cpu", "gpu")
invalid_devices = ("tpu",)

# Native data types
native_int8 = torch.int8
native_int16 = torch.int16
native_int32 = torch.int32
native_int64 = torch.int64
native_uint8 = torch.uint8
native_bfloat16 = torch.bfloat16
native_float16 = torch.float16
native_float32 = torch.float32
native_float64 = torch.float64
native_complex64 = torch.complex64
native_complex128 = torch.complex128
native_bool = torch.bool

# Valid and invalid data types
valid_dtypes = {
    "2.2 and below": (
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.bfloat16,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.bool,
    )
}

# Update valid_dtypes based on backend_version
valid_dtypes = _dtype_from_version(valid_dtypes, backend_version)

# Invalid data types
invalid_dtypes = {
    "2.2 and below": (
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
    )
}

# Update invalid_dtypes based on backend_version
invalid_dtypes = _dtype_from_version(invalid_dtypes, backend_version)

# Unsupported devices
unsupported_devices = ("tpu",)

native_inplace_support = True
supports_gradients = True

# Closest valid dtype function
def closest_valid_dtype(type=None, /, as_native=False):
    if type is None:
        type = ivy.default_dtype()
    elif isinstance(type, str) and type in invalid_dtypes:
        type = ivy.as_ivy_dtype({"uint16": ivy.uint8, "uint32": ivy.uint8, "uint64": ivy.uint8}[type])
    return ivy.as_ivy_dtype(type) if not as_native else ivy.as_native_dtype(type)

backend = "torch"

# Globals getter function
def globals_getter_func(x=None):
    if not x:
        return globals()
    else:
        globals()[x[0]] = x[1]

ivy.func_wrapper.globals_getter_func = globals_getter_func

# Import sub-modules
from . import activations
from . import creation
from . import data_type
from . import device
from . import elementwise
from . import gradients
from . import general
from . import layers
from . import linear_algebra as linalg
from . import manipulation
from . import random
from . import searching
from . import set
from . import sorting
from . import statistical
from . import utility
from . import experimental
from . import control_flow_ops
from . import norms
from . import module

# Import sub-backends
from . import sub_backends

# Native module
NativeModule = torch.nn.Module
