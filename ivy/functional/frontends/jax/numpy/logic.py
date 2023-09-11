# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.jax.numpy import (
    promote_types_of_jax_inputs as promote_jax_arrays,
)
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_supported_dtypes


def _packbits_nested_list_padding(arr, pad_length):
    if arr.ndim > 1:
        nested_list = []
        for sub_arr in arr:
            nested_list.append(_packbits_nested_list_padding(sub_arr, pad_length))
        return nested_list
    else:
        return arr.zero_pad(pad_width=[[0, pad_length]])


@to_ivy_arrays_and_back
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_jax_arrays(a, b)
    return ivy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@to_ivy_arrays_and_back
def array_equal(a1, a2, equal_nan: bool) -> bool:
    a1, a2 = promote_jax_arrays(a1, a2)
    if ivy.shape(a1) != ivy.shape(a2):
        return False
    eq = ivy.asarray(a1 == a2)
    if equal_nan:
        eq = ivy.logical_or(eq, ivy.logical_and(ivy.isnan(a1), ivy.isnan(a2)))
    return ivy.all(eq)


@to_ivy_arrays_and_back
def array_equiv(a1, a2) -> bool:
    a1, a2 = promote_jax_arrays(a1, a2)
    try:
        eq = ivy.equal(a1, a2)
    except ValueError:
        # shapes are not broadcastable
        return False
    return ivy.all(eq)


@to_ivy_arrays_and_back
def isneginf(x, /, out=None):
    return ivy.isinf(x, detect_positive=False, out=out)


@to_ivy_arrays_and_back
def isposinf(x, /, out=None):
    return ivy.isinf(x, detect_negative=False, out=out)


@to_ivy_arrays_and_back
def not_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.not_equal(x1, x2)


@to_ivy_arrays_and_back
def less(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.less(x1, x2)


@to_ivy_arrays_and_back
def less_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.less_equal(x1, x2)


@to_ivy_arrays_and_back
def greater(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.greater(x1, x2)


@to_ivy_arrays_and_back
def greater_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.greater_equal(x1, x2)


@to_ivy_arrays_and_back
def isnan(x, /):
    return ivy.isnan(x)


@to_ivy_arrays_and_back
def equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.equal(x1, x2)


@to_ivy_arrays_and_back
def all(a, axis=None, out=None, keepdims=False, *, where=False):
    return ivy.all(a, axis=axis, keepdims=keepdims, out=out)


@to_ivy_arrays_and_back
def bitwise_and(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.bitwise_and(x1, x2)


@to_ivy_arrays_and_back
def bitwise_not(x, /):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def bitwise_or(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.bitwise_or(x1, x2)


@to_ivy_arrays_and_back
def bitwise_xor(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.bitwise_xor(x1, x2)


@to_ivy_arrays_and_back
def any(a, axis=None, out=None, keepdims=False, *, where=None):
    # TODO: Out not supported
    ret = ivy.any(a, axis=axis, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(None, ivy.zeros_like(ret)))
    return ret


alltrue = all

sometrue = any
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_ivy_arrays_and_back
# known issue in jnp's documentation of arguments
# https://github.com/google/jax/issues/9119
def logical_and(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    if x1.dtype == "complex128" or x2.dtype == "complex128":
        x1 = ivy.astype(x1, ivy.complex128)
        x2 = ivy.astype(x2, ivy.complex128)
    else:
        x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.logical_and(x1, x2)


@to_ivy_arrays_and_back
def invert(x, /):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def isfinite(x, /):
    return ivy.isfinite(x)


@to_ivy_arrays_and_back
def isin(element, test_elements, assume_unique=False, invert=False):
    return ivy.isin(element, test_elements, assume_unique=assume_unique, invert=invert)


@to_ivy_arrays_and_back
def isinf(x, /):
    return ivy.isinf(x)


@to_ivy_arrays_and_back
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_jax_arrays(a, b)
    return ivy.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@to_ivy_arrays_and_back
def logical_not(x, /):
    return ivy.logical_not(x)


@to_ivy_arrays_and_back
def logical_or(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.logical_or(x1, x2)


@to_ivy_arrays_and_back
def isscalar(x, /):
    return ivy.isscalar(x)


@to_ivy_arrays_and_back
def left_shift(x1, x2):
    # TODO: implement
    raise IvyNotImplementedException()


@to_ivy_arrays_and_back
def isreal(x, out=None):
    return ivy.isreal(x, out=out)


@to_ivy_arrays_and_back
def logical_xor(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return ivy.logical_xor(x1, x2)


@to_ivy_arrays_and_back
def right_shift(x1, x2, /):
    return ivy.bitwise_right_shift(x1, x2)


@to_ivy_arrays_and_back
def isrealobj(x: any):
    return not ivy.is_complex_dtype(ivy.dtype(x))


@to_ivy_arrays_and_back
def iscomplex(x: any):
    return ivy.bitwise_invert(ivy.isreal(x))


@to_ivy_arrays_and_back
def iscomplexobj(x):
    return ivy.is_complex_dtype(ivy.dtype(x))


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.0.1 and below": ("int32", "int64", "float64", "float32")}, "torch"
)
@to_ivy_arrays_and_back
def setxor1d(ar1, ar2, assume_unique=False):
    common_dtype = ivy.promote_types(ivy.dtype(ar1), ivy.dtype(ar2))
    ar1 = ivy.asarray(ar1, dtype=common_dtype)
    ar2 = ivy.asarray(ar2, dtype=common_dtype)
    if not assume_unique:
        ar1 = ivy.unique_values(ar1)
        ar2 = ivy.unique_values(ar2)
    ar1 = ivy.reshape(ar1, (-1,))
    ar2 = ivy.reshape(ar2, (-1,))
    aux = ivy.concat([ar1, ar2], axis=0)
    if aux.size == 0:
        return aux
    aux = ivy.sort(aux)
    flag = ivy.concat(
        (ivy.array([True]), ivy.not_equal(aux[1:], aux[:-1]), ivy.array([True])), axis=0
    )
    mask = flag[1:] & flag[:-1]
    if ivy.all(ivy.logical_not(mask)):
        ret = ivy.asarray([], dtype=common_dtype)
    else:
        ret = aux[mask]
    return ret


@to_ivy_arrays_and_back
def packbits(x, /, *, axis=None, bitorder="big"):
    x = ivy.greater(x, ivy.zeros_like(x)).astype("uint8")
    bits = ivy.arange(8, dtype="uint8")
    if bitorder == "big":
        bits = bits[::-1]
    if axis is None:
        x = ivy.flatten(x)
        axis = 0
    x = ivy.swapaxes(x, axis, -1)

    remainder = x.shape[-1] % 8
    if remainder:
        x = _packbits_nested_list_padding(x, 8 - remainder)
        x = ivy.array(x)

    x = ivy.reshape(x, list(x.shape[:-1]) + [x.shape[-1] // 8, 8])
    bits = ivy.expand_dims(bits, axis=tuple(range(x.ndim - 1)))
    packed = (x << bits).sum(axis=-1).astype("uint8")
    return ivy.swapaxes(packed, axis, -1)
