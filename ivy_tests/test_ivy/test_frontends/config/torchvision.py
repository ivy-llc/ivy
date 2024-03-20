from .base import FrontendConfig, SupportedDtypes, SupportedDeviecs
import ivy


def get_config():
    return TorchVisionFrontendConfig()


class TorchVisionFrontendConfig(FrontendConfig):
    backend = ivy.with_backend("torch")

    valid_devices = ["cpu", "gpu"]
    invalid_devices = ["tpu"]

    valid_dtypes = [
        "int16",
        "int32",
        "int64",
        "uint8",
        "float16",
        "float32",
        "float64",
    ]

    invalid_dtypes = [
        "int8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "complex64",
        "complex128",
        "bool",
    ]

    valid_numeric_dtypes = [
        "int16",
        "int32",
        "int64",
        "uint8",
        "float16",
        "float32",
        "float64",
    ]

    invalid_numeric_dtypes = [
        "int8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "complex64",
        "complex128",
        "bool",
    ]

    valid_int_dtypes = [
        "int16",
        "int32",
        "int64",
        "uint8",
    ]

    invalid_int_dtypes = [
        "int8",
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
        "float16",
        "float32",
        "float64",
    ]

    invalid_float_dtypes = [
        "bfloat16",
    ]

    valid_complex_dtypes = []

    invalid_complex_dtypes = [
        "complex64",
        "complex128",
    ]

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

    @property
    def Dtype(self):
        return self.backend.Dtype

    @property
    def Device(self):
        return self.backend.Device

    def native_array(self, x):
        return self.backend.native_array(x)

    def is_native_array(self, x):
        return self.backend.is_native_array(x)

    def to_numpy(self, x):
        return self.backend.to_numpy(x)

    def as_native_dtype(self, dtype: str):
        return self.backend.as_native_dtype(dtype)

    def as_native_device(self, device: str):
        return self.backend.as_native_dev(device)

    def isscalar(self, x):
        return self.backend.isscalar(x)
