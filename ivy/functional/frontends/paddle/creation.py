# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
import ivy.functional.frontends.paddle as paddle_frontend
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def arange(start, end=None, step=1, dtype=None, name=None):
    return ivy.arange(start, end, step=step, dtype=dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64", "bool")},
    "paddle",
)
@to_ivy_arrays_and_back
def assign(x, output=None):
    if len(ivy.shape(x)) != 0:
        x = ivy.reshape(x, ivy.shape(x))
    ret = ivy.copy_array(x, to_ivy_array=False, out=output)
    return ret


@with_unsupported_dtypes(
    {"2.6.0 and below": ("bfloat16", "uint16", "uint32", "uint64")}, "paddle"
)
@to_ivy_arrays_and_back
def clone(x):
    return ivy.copy_array(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def complex(real, imag, name=None):
    assert real.dtype == imag.dtype, (
        "(InvalidArgument) The type of data we are trying to retrieve does not match"
        " the type of data currently contained in the container."
    )
    complex_dtype = "complex64" if real.dtype == "float32" else "complex128"
    imag_cmplx = ivy.astype(imag, complex_dtype) * 1j
    complex_array = real + imag_cmplx
    return complex_array


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def diag(x, offset=0, padding_value=0, name=None):
    if len(x.shape) == 1:
        padding_value = ivy.astype(padding_value, ivy.dtype(x))
        ret = ivy.diagflat(x, offset=offset, padding_value=padding_value)
        if len(ret.shape) != 2:
            ret = ivy.reshape(ret, (1, 1))
    else:
        ret = ivy.diag(x, k=offset)
    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def diagflat(x, offset=0, name=None):
    arr = ivy.diagflat(x, offset=offset)
    return arr


@to_ivy_arrays_and_back
def empty(shape, dtype=None):
    return ivy.empty(shape=shape, dtype=dtype)


@to_ivy_arrays_and_back
def empty_like(x, dtype=None, name=None):
    return ivy.empty_like(x, dtype=dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def eye(num_rows, num_columns=None, dtype=None, name=None):
    return ivy.eye(num_rows, num_columns, dtype=dtype)


@to_ivy_arrays_and_back
def full(shape, fill_value, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def full_like(x, fill_value, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.full_like(x, fill_value, dtype=dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def linspace(start, stop, num, dtype=None, name=None):
    return ivy.linspace(start, stop, num=num, dtype=dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def logspace(start, stop, num, base=10.0, dtype=None, name=None):
    return ivy.logspace(start, stop, num=num, base=base, dtype=dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def meshgrid(*args, **kwargs):
    return ivy.meshgrid(*args, indexing="ij")


@with_unsupported_dtypes({"2.6.0 and below": "int8"}, "paddle")
@to_ivy_arrays_and_back
def ones(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.ones(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def ones_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.ones_like(x, dtype=dtype)


@to_ivy_arrays_and_back
def to_tensor(data, /, *, dtype=None, place=None, stop_gradient=True):
    array = ivy.array(data, dtype=dtype, device=place)
    return paddle_frontend.Tensor(array, dtype=dtype, place=place)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "bool",
            "float64",
            "float32",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def tril(x, diagonal=0, name=None):
    return ivy.tril(x, k=diagonal)


@with_supported_dtypes({"2.6.0 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def tril_indices(row, col, offset=0, dtype="int64"):
    arr = ivy.tril_indices(row, col, offset)
    arr = ivy.astype(arr, dtype)
    return arr


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "float64",
            "float32",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def triu(x, diagonal=0, name=None):
    return ivy.triu(x, k=diagonal)


@with_supported_dtypes({"2.6.0 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def triu_indices(row, col=None, offset=0, dtype="int64"):
    arr = ivy.triu_indices(row, col, offset)
    if not ivy.to_scalar(ivy.shape(arr[0], as_array=True)):
        return arr
    arr = ivy.astype(arr, dtype)
    return arr


@with_unsupported_dtypes({"2.6.0 and below": "int8"}, "paddle")
@to_ivy_arrays_and_back
def zeros(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return ivy.zeros(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_ivy_arrays_and_back
def zeros_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return ivy.zeros_like(x, dtype=dtype)
