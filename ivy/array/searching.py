# global
import abc
from typing import Optional

# local
import ivy


# ToDo: implement all methods here as public instance methods


class ArrayWithSearching(abc.ABC):
    def argmax(
        self: ivy.Array,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> int:
        """
        ivy.Array instance method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        """
        return ivy.argmax(self, axis=axis, keepdims=keepdims, out=out)
