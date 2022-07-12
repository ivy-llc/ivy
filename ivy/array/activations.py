# global
import abc
from typing import Optional

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithActivations(abc.ABC):
    def softplus(
        self: ivy.Array,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softplus. This method simply wraps the
        function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.
        """
        return ivy.softplus(self._data, out=out)
