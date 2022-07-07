# global
import abc
from typing import Optional

# ToDo: implement all methods here as public instance methods
import ivy

class ArrayWithSet(abc.ABC):
    def unique_values(
            self: ivy.Array,
            out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.unique_values(
            self._data, out=out
        )

    def unique_counts(
            self: ivy.Array
    ) -> ivy.Array:
        return ivy.unique_counts(self._data)

    def unique_all(
            self: ivy.Array
    ) -> ivy.Array:
        return ivy.unique_all(self._data)
