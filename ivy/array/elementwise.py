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
