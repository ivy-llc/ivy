# import onnx
# import onnxruntime as ort

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


# Dtype = mx.numpy.dtype
# Device = mx.Context
#
#
# def native_array(x):
#     return mx.np.array(x)
#
#
# def is_native_array(x):
#     return isinstance(x, (mx.np.ndarray, mx.gluon.Parameter))
#
#
# def to_numpy(x):
#     return x.asnumpy()
#
#
# def as_native_dtype(dtype: str):
#     return mx.np.array([], dtype=dtype).dtype
#
#
# def as_native_dev(device: str):
#     return mx.Context(device)
#
#
# def isscalar(x):
#     return x.ndim == 0
#
#
