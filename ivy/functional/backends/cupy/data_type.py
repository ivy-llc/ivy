# global
import cupy as cp
from typing import Optional, Union, Sequence, List

# local
import ivy
from ivy.functional.ivy.data_type import _handle_nestable_dtype_info


ivy_dtype_dict = {
    cp.dtype("int8"): "int8",
    cp.dtype("int16"): "int16",
    cp.dtype("int32"): "int32",
    cp.dtype("int64"): "int64",
    cp.dtype("uint8"): "uint8",
    cp.dtype("uint16"): "uint16",
    cp.dtype("uint32"): "uint32",
    cp.dtype("uint64"): "uint64",
    cp.dtype("bfloat16"): "bfloat16",
    cp.dtype("float16"): "float16",
    cp.dtype("float32"): "float32",
    cp.dtype("float64"): "float64",
    cp.dtype("bool"): "bool",
    cp.int8: "int8",
    cp.int16: "int16",
    cp.int32: "int32",
    cp.int64: "int64",
    cp.uint8: "uint8",
    cp.uint16: "uint16",
    cp.uint32: "uint32",
    cp.uint64: "uint64",
    cp.float16: "float16",
    cp.float32: "float32",
    cp.float64: "float64",
    cp.bool_: "bool",
}

native_dtype_dict = {
    "int8": cp.dtype("int8"),
    "int16": cp.dtype("int16"),
    "int32": cp.dtype("int32"),
    "int64": cp.dtype("int64"),
    "uint8": cp.dtype("uint8"),
    "uint16": cp.dtype("uint16"),
    "uint32": cp.dtype("uint32"),
    "uint64": cp.dtype("uint64"),
    "float16": cp.dtype("float16"),
    "float32": cp.dtype("float32"),
    "float64": cp.dtype("float64"),
    "bool": cp.dtype("bool"),
}


class Finfo:
    def __init__(self, cp_finfo: cp.finfo):
        self._cp_finfo = cp_finfo

    def __repr__(self):
        return repr(self._cp_finfo)

    @property
    def bits(self):
        return self._cp_finfo.bits

    @property
    def eps(self):
        return float(self._cp_finfo.eps)

    @property
    def max(self):
        return float(self._cp_finfo.max)

    @property
    def min(self):
        return float(self._cp_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._cp_finfo.tiny)


# Array API Standard #
# -------------------#


def astype(
    x: cp.ndarray,
    dtype: cp.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[ivy.Array] = None,
) -> cp.ndarray:
    dtype = ivy.as_native_dtype(dtype)
    if copy:
        if x.dtype == dtype:
            new_tensor = cp.copy(x)
            return new_tensor
    else:
        if x.dtype == dtype:
            return x
        else:
            new_tensor = cp.copy(x)
            return new_tensor.astype(dtype)
    return x.astype(dtype)


def broadcast_arrays(*arrays: cp.ndarray) -> List[cp.ndarray]:
    return cp.broadcast_arrays(*arrays)


def broadcast_to(
    x: cp.ndarray, shape: Union[ivy.NativeShape, Sequence[int]]
) -> cp.ndarray:
    if x.ndim > len(shape):
        return cp.broadcast_to(x.reshape([-1]), shape)
    return cp.broadcast_to(x, shape)


def can_cast(from_: Union[cp.dtype, cp.ndarray], to: cp.dtype, /) -> bool:
    if isinstance(from_, cp.ndarray):
        from_ = str(from_.dtype)
    from_ = str(from_)
    to = str(to)
    if "bool" in from_ and (("int" in to) or ("float" in to)):
        return False
    if "int" in from_ and "float" in to:
        return False
    return cp.can_cast(from_, to)


@_handle_nestable_dtype_info
def finfo(type: Union[cp.dtype, str, cp.ndarray]) -> Finfo:
    if isinstance(type, cp.ndarray):
        type = type.dtype
    return Finfo(cp.finfo(ivy.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[cp.dtype, str, cp.ndarray]) -> cp.iinfo:
    if isinstance(type, cp.ndarray):
        type = type.dtype
    return cp.iinfo(ivy.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[cp.ndarray, cp.dtype]) -> ivy.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return cp.result_type(arrays_and_dtypes)
    result = cp.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = cp.result_type(result, arrays_and_dtypes[i])
    return as_ivy_dtype(result)


# Extra #
# ------#


def as_ivy_dtype(dtype_in: Union[cp.dtype, str]) -> ivy.Dtype:
    if isinstance(dtype_in, str):
        return ivy.Dtype(dtype_in)
    return ivy.Dtype(ivy_dtype_dict[dtype_in])


def as_native_dtype(dtype_in: Union[cp.dtype, str]) -> cp.dtype:
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict.values():
        return native_dtype_dict[ivy.Dtype(dtype_in)]
    else:
        raise ivy.exceptions.IvyException(
            f"Cannot convert to cupy dtype. {dtype_in} is not supported by CUPY."
        )


as_native_dtype.unsupported_dtypes = ("bfloat16",)


def dtype(x: cp.ndarray, as_native: bool = False) -> ivy.Dtype:
    if as_native:
        return ivy.to_native(x).dtype
    return as_ivy_dtype(x.dtype)


def dtype_bits(dtype_in: Union[cp.dtype, str]) -> int:
    dtype_str = as_ivy_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
    )
