import torch

valid_devices = ("cpu", "gpu")
invalid_devices = ("tpu",)

valid_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    # complex32 is available
    "complex64",
    "complex128",
    "bool",
]
invalid_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

valid_numeric_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
invalid_numeric_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

valid_int_dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
]
invalid_int_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

valid_uint_dtypes = [
    "uint8",
]
invalid_uint_dtypes = [
    "uint16",
    "uint32",
    "uint64",
]

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

dtype_dict = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# Helpers for function testing


Dtype = torch.dtype
Device = torch.device


def native_array(x):
    return torch.tensor(x)


def is_native_array(x):
    return isinstance(x, (torch.Tensor, torch.nn.Parameter))


def to_numpy(x):
    return x.numpy()


def as_native_dtype(dtype: str):
    return dtype_dict[dtype]


def as_native_dev(device: str):
    return torch.device(device)


def isscalar(x):
    return is_native_array(x) and x.dim() == 0
