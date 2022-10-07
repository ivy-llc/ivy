# global
import sys
import cupy as cp
# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

NativeArray = cp.ndarray
NativeVariable = cp.ndarray
NativeDevice = str
NativeDtype = cp.dtype
NativeShape = tuple

NativeSparseArray = None


# devices
valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")


# data types (preventing cyclic imports)
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.UintDtype("uint8")
uint16 = ivy.UintDtype("uint16")
uint32 = ivy.UintDtype("uint32")
uint64 = ivy.UintDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
bool = ivy.Dtype("bool")

# native data types
native_int8 = cp.dtype("int8")
native_int16 = cp.dtype("int16")
native_int32 = cp.dtype("int32")
native_int64 = cp.dtype("int64")
native_uint8 = cp.dtype("uint8")
native_uint16 = cp.dtype("uint16")
native_uint32 = cp.dtype("uint32")
native_uint64 = cp.dtype("uint64")
native_float16 = cp.dtype("float16")
native_float32 = cp.dtype("float32")
native_float64 = cp.dtype("float64")
native_double = native_float64
native_bool = cp.dtype("bool")

# valid data types
valid_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bool,
)
valid_numeric_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
)
valid_int_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
valid_float_dtypes = (float16, float32, float64)
valid_uint_dtypes = (uint8, uint16, uint32, uint64)

# invalid data types
invalid_dtypes = (bfloat16,)
invalid_numeric_dtypes = (bfloat16,)
invalid_int_dtypes = ()
invalid_float_dtypes = (bfloat16,)
invalid_uint_dtypes = ()

native_inplace_support = False

supports_gradients = False


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = ivy.as_ivy_dtype(type) # noqa
    if type_str in invalid_dtypes:
        return {"bfloat16": float16}[type_str]
    return type


backend = "cupy"
backend_version = cp.__version__

# local sub-modules
from . import activations # noqa: F401
from .activations import * # noqa: F401 F403
from . import compilation # noqa: F401
from .compilation import * # noqa: F401 F403
from . import creation # noqa: F401
from .creation import * # noqa: F401 F403
from . import data_type # noqa: F401
from .data_type import * # noqa: F401 F403
from . import device # noqa: F401
from .device import * # noqa: F401 F403
from . import elementwise # noqa: F401
from .elementwise import * # noqa: F401 F403
from . import extensions # noqa: F401
from .extensions import * # noqa: F401 F403
from . import general # noqa: F401
from .general import * # noqa: F401 F403
from . import gradients # noqa: F401
from .gradients import * # noqa: F401 F403
from . import layers # noqa: F401
from .layers import * # noqa: F401 F403
from . import linear_algebra as linalg
from .linear_algebra import * # noqa: F401 F403
from . import manipulation # noqa: F401
from .manipulation import * # noqa: F401 F403
from . import random # noqa: F401
from .random import * # noqa: F401 F403
from . import searching # noqa: F401
from .searching import * # noqa: F401 F403
from . import set # noqa: F401
from .set import * # noqa: F401 F403
from . import sorting # noqa: F401
from .sorting import * # noqa: F401 F403
from . import statistical # noqa: F401
from .statistical import * # noqa: F401 F403
from . import utility # noqa: F401
from .utility import * # noqa: F401 F403
