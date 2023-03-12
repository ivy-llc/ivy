from ivy.functional.frontends.jax.numpy import asarray
from ivy.functional.frontends.numpy import (
    dtype,
    generic,
    number,
    inexact,
    complexfloating,
    floating,
    integer,
    signedinteger,
    unsignedinteger,
)


class _ScalarMeta(type):
    def __hash__(self):
        return hash(self.dtype.type)

    def __eq__(self, other):
        return id(self) == id(other) or self.dtype.type == other

    def __ne__(self, other):
        return not (self == other)

    def __call__(self, x):
        return asarray(x, dtype=self.dtype)

    def __instancecheck__(self, instance):
        return isinstance(instance, self.dtype.type)


def _make_scalar_type(scalar_type):
    meta = _ScalarMeta(scalar_type, (object,), {"dtype": dtype(scalar_type)})
    return meta


bool_ = _make_scalar_type("bool_")
int8 = _make_scalar_type("int8")
int16 = _make_scalar_type("int16")
int32 = _make_scalar_type("int32")
int64 = _make_scalar_type("int64")
uint8 = _make_scalar_type("uint8")
uint16 = _make_scalar_type("uint16")
uint32 = _make_scalar_type("uint32")
uint64 = _make_scalar_type("uint64")
bfloat16 = _make_scalar_type("bfloat16")
float16 = _make_scalar_type("float16")
float32 = _make_scalar_type("float32")
float64 = _make_scalar_type("float64")
complex64 = _make_scalar_type("complex64")
complex128 = _make_scalar_type("complex128")

int_ = int64
uint = uint64
float_ = float64
compex_ = complex128

generic = generic
number = number
inexact = inexact
complexfloating = complexfloating
floating = floating
integer = integer
signedinteger = signedinteger
unsignedinteger = unsignedinteger
