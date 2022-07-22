# local
from typing import Union, Optional, Any

import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods

# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev(x: ivy.Container, as_native: bool = False) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.dev. This method simply
        wraps the function, and so the docstring for ivy.dev also applies to this
        method with minimal changes.

        """
        return ContainerBase.multi_map_in_static_method("dev", x, as_native=as_native)

    def dev(self: ivy.Container, as_native: bool = False) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.dev. This method simply
        wraps the function, and so the docstring for ivy.dev also applies to this
        method with minimal changes.

        """
        return self.static_dev(self, as_native=as_native)

    @staticmethod
    def static_to_device(
        x: ivy.Container,
        device: Union[ivy.Device, ivy.NativeDevice],
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_device. This method
        simply wraps the function, and so the docstring for ivy.to_device also
        applies to this method with minimal changes.

        """
        return ContainerBase.multi_map_in_static_method(
            "to_device", x, device, stream=stream, out=out
        )

    def to_device(
        self: ivy.Container,
        device: Union[ivy.Device, ivy.NativeDevice],
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_device. This method
        simply wraps the function, and so the docstring for ivy.to_device also
        applies to this method with minimal changes.

        """
        return self.static_to_device(self, device, stream=stream, out=out)
