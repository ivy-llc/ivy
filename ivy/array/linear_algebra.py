# global
import abc
from typing import Union, Optional

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithLinearAlgebra(abc.ABC):
    def matmul(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.matmul(self, x2, out=out)
