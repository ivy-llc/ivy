# global
import abc
from typing import Optional, Union

# local
import ivy


class ArrayWithLosses(abc.ABC):
    def cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.cross_entropy(self._data, pred, axis=axis, epsilon=epsilon, out=out)

    def binary_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.binary_cross_entropy(self._data, pred, epsilon=epsilon, out=out)

    def sparse_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        epsilon: float = 1e-7,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.sparse_cross_entropy(
            self._data, pred, axis=axis, epsilon=epsilon, out=out
        )
