# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):
    def abs(
        self: ivy.Array, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
    ) -> ivy.Array:
        return ivy.abs(self, out=out)

    def acosh(
        self: ivy.Array, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
    ) -> ivy.Array:
        return ivy.acosh(self, out=out)

    def acos(
        self: ivy.Array, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
    ) -> ivy.Array:
        return ivy.acos(self, out=out)

    def add(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.add(self, x2, out=out)

    def asin(
        self: ivy.Array, out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
    ) -> ivy.Array:
        return ivy.asin(self, out=out)
