# global
import numpy as np
from typing import Optional, Union, Sequence, List

# local
import ivy


ivy_dtype_dict = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    "bfloat16": "bfloat16",
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.bool_: "bool",
}

native_dtype_dict = {
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "bool": np.dtype("bool"),
}


class Finfo:
    def __init__(self, np_finfo: np.finfo):
        self._np_finfo = np_finfo

    def __repr__(self):
        return repr(self._np_finfo)

    @property
    def bits(self):
        return self._np_finfo.bits

    @property
    def eps(self):
        return float(self._np_finfo.eps)

    @property
    def max(self):
        return float(self._np_finfo.max)

    @property
    def min(self):
        return float(self._np_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._np_finfo.tiny)


# Array API Standard #
# -------------------#


def astype(
        x: np.ndarray, dtype: np.dtype, 
        *, 
        copy: bool = True, 
        out: Optional[ivy.Array] = None,) -> np.ndarray:
    dtype = ivy.as_native_dtype(dtype)
    if copy:
        if x.dtype == dtype:
            new_tensor = np.copy(x)
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = np.copy(x)
            return new_tensor.astype(dtype)
    return x.astype(dtype)


def broadcast_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    return np.broadcast_arrays(*arrays)


def broadcast_to(
    x: np.ndarray, shape: Union[ivy.NativeShape, Sequence[int]]
) -> np.ndarray:
    return np.broadcast_to(x, shape)


def can_cast(from_: Union[np.dtype, np.ndarray], to: np.dtype) -> bool:
    if isinstance(from_, np.ndarray):
        from_ = str(from_.dtype)
    from_ = str(from_)
    to = str(to)
    if "bool" in from_ and (("int" in to) or ("float" in to)):
        return False
    if "int" in from_ and "float" in to:
        return False
    return np.can_cast(from_, to)


def finfo(type: Union[np.dtype, str, np.ndarray]) -> Finfo:
    if isinstance(type, np.ndarray):
        type = type.dtype
    return Finfo(np.finfo(ivy.as_native_dtype(type)))


def iinfo(type: Union[np.dtype, str, np.ndarray]) -> np.iinfo:
    if isinstance(type, np.ndarray):
        type = type.dtype
    return np.iinfo(ivy.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[np.ndarray, np.dtype]) -> ivy.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return np.result_type(arrays_and_dtypes)
    result = np.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = np.result_type(result, arrays_and_dtypes[i])
    return as_ivy_dtype(result)


# Extra #
# ------#


def as_ivy_dtype(dtype_in: Union[np.dtype, str]) -> ivy.Dtype:
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[np.dtype, str]) -> np.dtype:
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.values():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise TypeError(
            f"Cannot convert to numpy dtype. {dtype_in} is not supported by NumPy."
        )


as_native_dtype.unsupported_dtypes = ("bfloat16",)


def dtype(x: np.ndarray, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[np.dtype, str]) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )
