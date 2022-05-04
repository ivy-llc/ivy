# global
import abc
from typing import Optional, Union, Tuple, List

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithManipulation(abc.ABC):
    def concat(
        self: ivy.Array,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        out: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.concat([self] + xs, axis, out=out)
