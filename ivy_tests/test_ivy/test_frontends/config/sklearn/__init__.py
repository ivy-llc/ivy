from ..base import FrontendConfig, SupportedDeviecs, SupportedDtypes
import numpy as np


def get_config():
    return SklearnFrontendConfig()


class SklearnFrontendConfig(FrontendConfig):
    Dtype = np.dtype
    Device = str

    valid_devices = "cpu"
    invalid_devices = ("gpu", "tpu")

    valid_dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
    ]
    invalid_dtypes = ["bfloat16"]

    valid_numeric_dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
    invalid_numeric_dtypes = ["bfloat16"]

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
        "float16",
        "float32",
        "float64",
    ]
    invalid_float_dtypes = ["bfloat16"]

    valid_complex_dtypes = [
        "complex64",
        "complex128",
    ]
    invalid_complex_dtypes = []

    # todo: add extension types

    @property
    def supported_devices(self):
        return SupportedDeviecs(
            valid_devices=self.valid_devices, invalid_devices=self.invalid_devices
        )

    @property
    def supported_dtypes(self):
        return SupportedDtypes(
            valid_dtypes=self.valid_dtypes,
            invalid_dtypes=self.invalid_dtypes,
            valid_numeric_dtypes=self.valid_numeric_dtypes,
            invalid_numeric_dtypes=self.invalid_numeric_dtypes,
            valid_int_dtypes=self.valid_int_dtypes,
            invalid_int_dtypes=self.invalid_int_dtypes,
            valid_uint_dtypes=self.valid_uint_dtypes,
            invalid_uint_dtypes=self.invalid_uint_dtypes,
            valid_float_dtypes=self.valid_float_dtypes,
            invalid_float_dtypes=self.invalid_float_dtypes,
            valid_complex_dtypes=self.valid_complex_dtypes,
            invalid_complex_dtypes=self.invalid_complex_dtypes,
        )

    def native_array(self, x):
        return np.array(x)

    def is_native_array(self, x):
        return isinstance(x, np.ndarray)

    def to_numpy(self, x):
        if not isinstance(x, np.ndarray):
            return np.array(x)
        return x

    def as_native_dtype(self, dtype: str):
        return np.dtype(dtype)

    def as_native_device(self, device: str):
        return device

    def isscalar(self, x):
        return np.isscalar(x)
