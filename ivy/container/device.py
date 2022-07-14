# local
from typing import Union

import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods

# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev(container: ivy.Container) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.device.dev. This method simply
        wraps the function, and so the docstring for ivy.device.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return ContainerBase.multi_map_in_static_method("dev", container)

    def dev(self: ivy.Container) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.dev. This method simply
        wraps the function, and so the docstring for ivy.device.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return self.static_dev(self)
