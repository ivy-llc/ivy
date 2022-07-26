# global
from typing import Optional

# local
import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSearching(ContainerBase):
    @staticmethod
    def static_argmax(
        x: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        """
        return ContainerBase.multi_map_in_static_method(
            "argmax", x, axis=axis, keepdims=keepdims, out=out
        )

    def argmax(
        self: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        """
        return self.static_argmax(self, axis=axis, keepdims=keepdims, out=out)
