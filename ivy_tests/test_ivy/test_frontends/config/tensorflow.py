from .base import FrontendConfig, SupportedDeviecs, SupportedDtypes
import ivy


def get_config():
    return TensorflowFrontendConfig()


class TensorflowFrontendConfig(FrontendConfig):
    def __init__(self):
        self.backend = ivy.with_backend("tensorflow", cached=True)

    @property
    def Dtype(self):
        return self.backend.Dtype

    @property
    def Device(self):
        return self.backend.Device

    @property
    def supported_devices(self):
        return SupportedDeviecs(
            valid_devices=self.backend.valid_devices,
            invalid_devices=self.backend.invalid_devices,
        )

    @property
    def supported_dtypes(self):
        return SupportedDtypes(
            valid_dtypes=self.backend.valid_dtypes,
            invalid_dtypes=self.backend.invalid_dtypes,
            valid_numeric_dtypes=self.backend.valid_numeric_dtypes,
            invalid_numeric_dtypes=self.backend.invalid_numeric_dtypes,
            valid_int_dtypes=self.backend.valid_int_dtypes,
            invalid_int_dtypes=self.backend.invalid_int_dtypes,
            valid_uint_dtypes=self.backend.valid_uint_dtypes,
            invalid_uint_dtypes=self.backend.invalid_uint_dtypes,
            valid_float_dtypes=self.backend.valid_float_dtypes,
            invalid_float_dtypes=self.backend.invalid_float_dtypes,
            valid_complex_dtypes=self.backend.valid_complex_dtypes,
            invalid_complex_dtypes=self.backend.invalid_complex_dtypes,
        )

    def native_array(self, x):
        return self.backend.native_array(x)

    def is_native_array(self, x):
        return self.backend.is_native_array(x)

    def to_numpy(self, x):
        return self.backend.to_numpy(x)

    def as_native_dtype(self, dtype: str):
        return self.backend.as_native_dtype(dtype)

    def as_native_device(self, device: str):
        return self.backend_as_native_dev(device)

    def isscalar(self, x):
        return self.backend.isscalar(x)
