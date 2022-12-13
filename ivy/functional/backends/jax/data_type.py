# global
import numpy as np

import jax.numpy as jnp
from typing import Union, Sequence, List

# local
import ivy
from ivy.functional.backends.jax import JaxArray
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info

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


def astype(x: JaxArray, dtype: jnp.dtype, /, *, copy: bool = True) -> JaxArray:
    dtype = ivy.as_native_dtype(dtype)
    if x.dtype == dtype:
        return jnp.copy(x) if copy else x
    return x.astype(dtype)


def broadcast_arrays(*arrays: JaxArray) -> List[JaxArray]:
    return jnp.broadcast_arrays(*arrays)


def broadcast_to(x: JaxArray, shape: Union[ivy.NativeShape, Sequence[int]]) -> JaxArray:
    if x.ndim > len(shape):
        return jnp.broadcast_to(x.reshape(-1), shape)
    return jnp.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[jnp.dtype, str, JaxArray]) -> Finfo:
    return Finfo(jnp.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[jnp.dtype, str, JaxArray]) -> np.iinfo:
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


def as_ivy_dtype(dtype_in: Union[jnp.dtype, str, bool, int, float]) -> ivy.Dtype:
    if dtype_in is int:
        return ivy.default_int_dtype()
    if dtype_in is float:
        return ivy.default_float_dtype()
    if dtype_in is bool:
        return ivy.Dtype("bool")
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            return ivy.Dtype(dtype_in)
        else:
            raise ivy.exceptions.IvyException(
                "Cannot convert to ivy dtype."
                f" {dtype_in} is not supported by Jax backend."
            )
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[jnp.dtype, str, bool, int, float]) -> jnp.dtype:
    if dtype_in is int:
        return ivy.default_int_dtype(as_native=True)
    if dtype_in is float:
        return ivy.default_float_dtype(as_native=True)
    if dtype_in is bool:
        return jnp.dtype("bool")
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.values():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.exceptions.IvyException(
            f"Cannot convert to Jax dtype. {dtype_in} is not supported by Jax."
        )


def dtype(x: JaxArray, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[jnp.dtype, str]) -> int:
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
