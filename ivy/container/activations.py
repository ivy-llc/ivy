# global
from typing import Optional, Union

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithActivations(ContainerBase):
    def softplus(
        self: ivy.Container,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softplus. 
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.
        """
        return ContainerWithActivations.static_softplus(self, out=out)
    
    @staticmethod
    def static_softplus(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softplus.
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "softplus",
            x,
            out=out
        )
