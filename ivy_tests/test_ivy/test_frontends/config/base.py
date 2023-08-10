from abc import ABC, abstractproperty, abstractmethod
from dataclasses import dataclass
from typing import List


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
