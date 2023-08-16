from abc import ABC, abstractproperty, abstractmethod
from dataclasses import dataclass
from typing import List
import ivy


@dataclass
class SupportedDeviecs:
    valid_devices: List[str]
    invalid_devices: List[str]


# TODO can be refactored and be constructed dynamically
@dataclass
class SupportedDtypes:
    valid_dtypes: List[str]
    invalid_dtypes: List[str]

    valid_numeric_dtypes: List[str]
    invalid_numeric_dtypes: List[str]

    valid_int_dtypes: List[str]
    invalid_int_dtypes: List[str]

    valid_uint_dtypes: List[str]
    invalid_uint_dtypes: List[str]

    valid_uint_dtypes: List[str]
    invalid_uint_dtypes: List[str]

    valid_float_dtypes: List[str]
    invalid_float_dtypes: List[str]

    valid_complex_dtypes: List[str]
    invalid_complex_dtypes: List[str]


class FrontendConfig(ABC):
    @abstractproperty
    def supported_dtypes(self) -> SupportedDtypes:
        pass

    @abstractproperty
    def supported_devices(self) -> SupportedDeviecs:
        pass

    @abstractproperty
    def Dtype(self):
        pass

    @abstractproperty
    def Device(self):
        pass

    @abstractmethod
    def native_array(self, x):
        pass

    @abstractmethod
    def is_native_array(self, x):
        pass

    @abstractmethod
    def to_numpy(self, x):
        pass

    @abstractmethod
    def as_native_dtype(self, dtype: str):
        pass

    @abstractmethod
    def as_native_device(self, device: str):
        pass

    @abstractmethod
    def isscalar(self, x):
        pass


class FrontendConfigWithBackend(FrontendConfig):
    backend_str = None

    def __init__(self):
        self.backend = ivy.with_backend(self.backend_str, cached=True)

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
