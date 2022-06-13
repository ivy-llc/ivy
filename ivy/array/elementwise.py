# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):
    def abs(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.abs(self._data, out=out)

    def acosh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acosh(self._data, out=out)

    def acos(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acos(self._data, out=out)

    def add(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.add(self._data, x2, out=out)

    def asin(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.asin(self._data, out=out)

    def asinh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.asinh(self._data, out=out)

    def atan(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atan(self._data, out=out)

    def atan2(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.atan2(self._data, x2, out=out)

    def atanh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atanh(self._data, out=out)

    def bitwise_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_and(self._data, x2, out=out)

    def bitwise_left_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_left_shift(self._data, x2, out=out)

    def bitwise_invert(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.bitwise_invert(self._data, out=out)

    def bitwise_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_or(self._data, x2, out=out)

    def bitwise_right_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_right_shift(self._data, x2, out=out)

    def bitwise_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.bitwise_xor(self._data, x2, out=out)

    def ceil(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.ceil(self._data, out=out)

    def cos(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cos(self._data, out=out)

    def cosh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cosh(self._data, out=out)

    def divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.divide(self._data, x2, out=out)

    def equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.equal(self._data, x2, out=out)

    def exp(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.exp(self._data, out=out)

    def expm1(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.expm1(self._data, out=out)

    def floor(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.floor(self._data, out=out)

    def floor_divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.floor_divide(self._data, x2, out=out)

    def greater(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.greater(self._data, x2, out=out)

    def greater_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.greater_equal(self._data, x2, out=out)

    def isfinite(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isfinite(self._data, out=out)

    def isinf(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isinf(self._data, out=out)

    def isnan(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isnan(self._data, out=out)

    def less(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.less(self._data, x2, out=out)

    def less_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.less_equal(self._data, x2, out=out)

    def log(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log(self._data, out=out)

    def log1p(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log1p(self._data, out=out)

    def log2(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log2(self._data, out=out)

    def log10(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log10(self._data, out=out)

    def logaddexp(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logaddexp(self._data, x2, out=out)

    def logical_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_and(self._data, x2, out=out)

    def logical_not(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.logical_not(self._data, out=out)

    def logical_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_or(self._data, x2, out=out)

    def logical_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.logical_xor(self._data, x2, out=out)

    def multiply(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.multiply(self._data, x2, out=out)

    def negative(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.negative(self._data, out=out)

    def not_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.not_equal(self._data, x2, out=out)

    def positive(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.positive(self._data, out=out)

    def pow(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.pow(self._data, x2, out=out)

    def remainder(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.remainder(self._data, x2, out=out)

    def round(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.round(self._data, out=out)

    def sign(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sign(self._data, out=out)

    def sin(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sin(self._data, out=out)

    def sinh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sinh(self._data, out=out)

    def square(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.square(self._data, out=out)

    def sqrt(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sqrt(self._data, out=out)

    def subtract(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.subtract(self._data, x2, out=out)

    def tan(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tan(self._data, out=out)

    def tanh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tanh(self._data, out=out)

    def trunc(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.trunc(self._data, out=out)

    def erf(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.erf(self._data, out=out)
