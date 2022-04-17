# global
import abc
from typing import Optional, Union

# ToDo: implement all methods here as public instance methods

# local
import ivy


# noinspection PyUnresolvedReferences
class ArrayWithElementwise(abc.ABC):

    def abs(self,
            out: Optional[Union[ivy.Array, ivy.NativeArray]] = None)\
            -> ivy.Array:
        return ivy.abs(self._data, out=out)
