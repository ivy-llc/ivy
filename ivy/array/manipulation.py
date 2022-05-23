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
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.concat([self] + xs, axis, out=out)

    def flip(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Array:
        return ivy.flip(self, axis, out=out)

    def expand_dims(
        self: ivy.Array,
        axis: Optional[int] = 0,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.expand_dims(self, axis, out=out)

    def reshape(
        self: ivy.Array,
        shape: Tuple[int, ...],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.reshape(self, shape, out=out)

    def permute_dims(
        self: ivy.Array,
        axes: Tuple[int, ...],
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> Union[ivy.Array, ivy.Container]:
        return ivy.permute_dims(self, axes, out=out)
