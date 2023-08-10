from .base import FrontendConfig, SupportedDeviecs, SupportedDtypes
import torch


class TorchFrontendConfig(FrontendConfig):
    Dtype = torch.dtype
    Device = torch.device

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
        return torch.tensor(x)

    def is_native_array(self, x):
        return isinstance(x, (torch.Tensor, torch.nn.Parameter))

    def to_numpy(self, x):
        return x.numpy()

    def as_native_dtype(self, dtype: str):
        return self.dtype_dict[dtype]

    def as_native_device(self, device: str):
        return torch.device(device)

    def isscalar(self, x):
        return self.is_native_array(x) and x.dim() == 0
