import jax.numpy as jnp
import jax
import numpy as np

valid_devices = ("cpu", "gpu")
invalid_devices = ("tpu",)

valid_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]
invalid_dtypes = []

valid_numeric_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
invalid_numeric_dtypes = []

valid_int_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
invalid_int_dtypes = []

valid_uint_dtypes = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
invalid_uint_dtypes = []

valid_float_dtypes = [
    "bfloat16",
    "float16",
    "float32",
    "float64",
]
invalid_float_dtypes = []

valid_complex_dtypes = [
    "complex64",
    "complex128",
]
invalid_complex_dtypes = []


# Helpers for function testing


Dtype = jax.dtypes.DType
Device = jax.devices.Device


def native_array(x):
    return jnp.array(x)


def is_native_array(x):
    return isinstance(x, (jnp.ndarray, jnp.DeviceArray))


def to_numpy(x):
    return np.asarray(x)


def as_native_dtype(dtype: str):
    return jnp.dtype(dtype)


def as_native_dev(device: str):
    return jax.devices(device)[0]


def isscalar(x):
    return x.ndim == 0
