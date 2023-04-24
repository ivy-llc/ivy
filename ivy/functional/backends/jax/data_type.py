# global
import numpy as np

import jax.numpy as jnp
from typing import Optional, Union, Sequence, List

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

ivy_dtype_dict = {
    jnp.dtype("int8"): "int8",
    jnp.dtype("int16"): "int16",
    jnp.dtype("int32"): "int32",
    jnp.dtype("int64"): "int64",
    jnp.dtype("uint8"): "uint8",
    jnp.dtype("uint16"): "uint16",
    jnp.dtype("uint32"): "uint32",
    jnp.dtype("uint64"): "uint64",
    jnp.dtype("bfloat16"): "bfloat16",
    jnp.dtype("float16"): "float16",
    jnp.dtype("float32"): "float32",
    jnp.dtype("float64"): "float64",
    jnp.dtype("complex64"): "complex64",
    jnp.dtype("complex128"): "complex128",
    jnp.dtype("bool"): "bool",
    jnp.int8: "int8",
    jnp.int16: "int16",
    jnp.int32: "int32",
    jnp.int64: "int64",
    jnp.uint8: "uint8",
    jnp.uint16: "uint16",
    jnp.uint32: "uint32",
    jnp.uint64: "uint64",
    jnp.bfloat16: "bfloat16",
    jnp.float16: "float16",
    jnp.float32: "float32",
    jnp.float64: "float64",
    jnp.complex64: "complex64",
    jnp.complex128: "complex128",
    jnp.bool_: "bool",
}

native_dtype_dict = {
    "int8": jnp.dtype("int8"),
    "int16": jnp.dtype("int16"),
    "int32": jnp.dtype("int32"),
    "int64": jnp.dtype("int64"),
    "uint8": jnp.dtype("uint8"),
    "uint16": jnp.dtype("uint16"),
    "uint32": jnp.dtype("uint32"),
    "uint64": jnp.dtype("uint64"),
    "bfloat16": jnp.dtype("bfloat16"),
    "float16": jnp.dtype("float16"),
    "float32": jnp.dtype("float32"),
    "float64": jnp.dtype("float64"),
    "complex64": jnp.dtype("complex64"),
    "complex128": jnp.dtype("complex128"),
    "bool": jnp.dtype("bool"),
}

char_rep_dtype_dict = {
    "?": "bool",
    "i": int,
    "i1": "int8",
    "i2": "int16",
    "i4": "int32",
    "i8": "int64",
    "f": float,
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "c": complex,
    "c8": "complex64",
    "c16": "complex128",
    "u": "uint32",
    "u1": "uint8",
    "u2": "uint16",
    "u4": "uint32",
    "u8": "uint64",
}


class Finfo:
    def __init__(self, jnp_finfo: jnp.finfo):
        self._jnp_finfo = jnp_finfo

    def __repr__(self):
        return repr(self._jnp_finfo)

    @property
    def bits(self):
        return self._jnp_finfo.bits

    @property
    def eps(self):
        return float(self._jnp_finfo.eps)

    @property
    def max(self):
        return float(self._jnp_finfo.max)

    @property
    def min(self):
        return float(self._jnp_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._jnp_finfo.tiny)


# Array API Standard #
# -------------------#


def astype(
    x: JaxArray,
    dtype: jnp.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    ivy.utils.assertions._check_jax_x64_flag(dtype)
    if x.dtype == dtype:
        return jnp.copy(x) if copy else x
    return x.astype(dtype)


def broadcast_arrays(*arrays: JaxArray) -> List[JaxArray]:
    return jnp.broadcast_arrays(*arrays)


@with_unsupported_dtypes(
    {"0.3.14 and below": ("complex")},
    backend_version,
)
def broadcast_to(
    x: JaxArray,
    /,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if x.ndim > len(shape):
        return jnp.broadcast_to(x.reshape(-1), shape)
    return jnp.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[jnp.dtype, str, JaxArray, np.ndarray], /) -> Finfo:
    if isinstance(type, np.ndarray):
        type = type.dtype.name
    return Finfo(jnp.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[jnp.dtype, str, JaxArray, np.ndarray], /) -> np.iinfo:
    if isinstance(type, np.ndarray):
        type = type.dtype.name
    return jnp.iinfo(ivy.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[JaxArray, jnp.dtype]) -> ivy.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return jnp.result_type(arrays_and_dtypes)

    result = jnp.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = jnp.result_type(result, arrays_and_dtypes[i])
    return as_ivy_dtype(result)


# Extra #
# ------#


def as_ivy_dtype(
    dtype_in: Union[jnp.dtype, str, int, float, complex, bool, np.dtype],
    /,
) -> ivy.Dtype:
    if dtype_in is int:
        return ivy.default_int_dtype()
    if dtype_in is float:
        return ivy.default_float_dtype()
    if dtype_in is complex:
        return ivy.default_complex_dtype()
    if dtype_in is bool:
        return ivy.Dtype("bool")
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if isinstance(dtype_in, str):
        if dtype_in in char_rep_dtype_dict:
            return as_ivy_dtype(char_rep_dtype_dict[dtype_in])
        if dtype_in in native_dtype_dict:
            dtype_str = dtype_in
        else:
            raise ivy.utils.exceptions.IvyException(
                "Cannot convert to ivy dtype."
                f" {dtype_in} is not supported by JAX backend."
            )
    else:
        dtype_str = ivy_dtype_dict[dtype_in]

    if "uint" in dtype_str:
        return ivy.UintDtype(dtype_str)
    elif "int" in dtype_str:
        return ivy.IntDtype(dtype_str)
    elif "float" in dtype_str:
        return ivy.FloatDtype(dtype_str)
    elif "complex" in dtype_str:
        return ivy.ComplexDtype(dtype_str)
    elif "bool" in dtype_str:
        return ivy.Dtype("bool")
    else:
        raise ivy.utils.exceptions.IvyException(
            f"Cannot recognize {dtype_str} as a valid Dtype."
        )


def as_native_dtype(
    dtype_in: Union[jnp.dtype, str, bool, int, float, np.dtype],
) -> jnp.dtype:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return ivy.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return jnp.dtype("bool")
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in char_rep_dtype_dict:
        return as_native_dtype(char_rep_dtype_dict[dtype_in])
    if dtype_in in native_dtype_dict.values():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.utils.exceptions.IvyException(
            f"Cannot convert to Jax dtype. {dtype_in} is not supported by Jax."
        )


def dtype(x: Union[JaxArray, np.ndarray], *, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.as_native_dtype(x.dtype)
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[jnp.dtype, str, np.dtype], /) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )


def is_native_dtype(dtype_in: Union[jnp.dtype, str], /) -> bool:
    if dtype_in in ivy_dtype_dict:
        return True
    else:
        return False
