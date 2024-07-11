from . import tree
import ivy
from ivy.functional.frontends.numpy import array

_int8 = ivy.IntDtype("int8")
_int16 = ivy.IntDtype("int16")
_int32 = ivy.IntDtype("int32")
_int64 = ivy.IntDtype("int64")
_uint8 = ivy.UintDtype("uint8")
_uint16 = ivy.UintDtype("uint16")
_uint32 = ivy.UintDtype("uint32")
_uint64 = ivy.UintDtype("uint64")
_bfloat16 = ivy.FloatDtype("bfloat16")
_float16 = ivy.FloatDtype("float16")
_float32 = ivy.FloatDtype("float32")
_float64 = ivy.FloatDtype("float64")
_complex64 = ivy.ComplexDtype("complex64")
_complex128 = ivy.ComplexDtype("complex128")
_bool = ivy.Dtype("bool")

_frontend_array = array
