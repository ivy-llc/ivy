# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):
    def abs(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.abs(self, out=out)

    def acosh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acosh(self, out=out)

    def acos(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.acos(self, out=out)

    def add(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.add(self, x2, out=out)

    def asin(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.asin(self, out=out)

    def asinh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.asinh(self, out=out)

    def atan(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atan(self, out=out)

    def atan2(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.atan2(self, x2, out=out)

    def atanh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.atanh(self, out=out)

    def bitwise_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.bitwise_and(self, x2, out=out)

    def bitwise_left_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.bitwise_left_shift(self, x2, out=out)

    def bitwise_invert(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.bitwise_invert(self, out=out)

    def bitwise_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.bitwise_or(self, x2, out=out)

    def bitwise_right_shift(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.bitwise_right_shift(self, x2, out=out)

    def bitwise_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.bitwise_xor(self, x2, out=out)

    def ceil(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.ceil(self, out=out)

    def cos(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cos(self, out=out)

    def cosh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.cosh(self, out=out)

    def divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.divide(self, x2, out=out)

    def equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.equal(self, x2, out=out)

    def exp(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.exp(self, out=out)

    def expm1(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.expm1(self, out=out)

    def floor(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.floor(self, out=out)

    def floor_divide(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.floor_divide(self, x2, out=out)

    def greater(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.greater(self, x2, out=out)

    def greater_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.greater_equal(self, x2, out=out)

    def isfinite(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isfinite(self, out=out)

    def isinf(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isinf(self, out=out)

    def isnan(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.isnan(self, out=out)

    def less(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.less(self, x2, out=out)

    def less_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.less_equal(self, x2, out=out)

    def log(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log(self, out=out)

    def log1p(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log1p(self, out=out)

    def log2(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log2(self, out=out)

    def log10(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.log10(self, out=out)

    def logaddexp(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.logaddexp(self, x2, out=out)

    def logical_and(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.logical_and(self, x2, out=out)

    def logical_not(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.logical_not(self, out=out)

    def logical_or(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.logical_or(self, x2, out=out)

    def logical_xor(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.logical_xor(self, x2, out=out)

    def multiply(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.multiply(self, x2, out=out)

    def negative(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.negative(self, out=out)

    def not_equal(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.not_equal(self, x2, out=out)

    def positive(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.positive(self, out=out)

    def pow(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.pow(self, x2, out=out)

    def remainder(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.remainder(self, x2, out=out)

    def round(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.round(self, out=out)

    def sign(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sign(self, out=out)

    def sin(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sin(self, out=out)

    def sinh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sinh(self, out=out)

    def square(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.square(self, out=out)

    def sqrt(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.sqrt(self, out=out)

    def subtract(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.subtract(self, x2, out=out)

    def tan(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tan(self, out=out)

    def tanh(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tanh(self, out=out)

    def trunc(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.trunc(self, out=out)

    def erf(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.erf(self, out=out)
