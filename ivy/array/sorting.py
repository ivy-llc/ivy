# global
import abc
from typing import Optional
import ivy


class ArrayWithSorting(abc.ABC):
    def argsort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.argsort(self._data, axis, descending, stable, out=out)

    def sort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.sort(self._data, axis, descending, stable, out=out)
