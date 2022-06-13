# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithLosses(abc.ABC):
    def cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[int] = -1,
        epsilon: Optional[float] = 1e-7,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.cross_entropy(self, pred, axis=axis, epsilon=epsilon, out=out)

    def binary_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        epsilon: Optional[float] = 1e-7,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.binary_cross_entropy(self, pred, epsilon=epsilon, out=out)

    pass
