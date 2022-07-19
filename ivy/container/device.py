# local
from typing import Union, Optional, Any, Iterable

import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods

# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev(x: ivy.Container) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.device.dev. This method simply
        wraps the function, and so the docstring for ivy.device.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return ContainerBase.multi_map_in_static_method("dev", x)

    def dev(self: ivy.Container) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.dev. This method simply
        wraps the function, and so the docstring for ivy.device.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return self.static_dev(self)

    @staticmethod
    def static_to_device(
        x: ivy.Container,
        device: Union[ivy.Device, ivy.NativeDevice],
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.to_device. This method
        simply wraps the function, and so the docstring for ivy.device.to_device also
        applies to this method with minimal changes.

        Examples
        --------

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
        ivy.Container instance method variant of ivy.device.to_device. This method
        simply wraps the function, and so the docstring for ivy.device.to_device also
        applies to this method with minimal changes.

        Examples
        --------

        """
        return self.static_to_device(self, device, stream=stream, out=out)

    @staticmethod
    def static_dev_clone_array(
        x: ivy.Container, devices: Iterable[ivy.Device]
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.dev_clone_array. This method
        simply wraps the function, and so the docstring for ivy.device.dev_clone_array
        also applies to this method with minimal changes.

        Examples
        --------

        """
        return ContainerBase.multi_map_in_static_method("dev_clone_array", x, devices)

    def dev_clone_array(
        self: ivy.Container, devices: Iterable[ivy.Device]
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.dev_clone_array. This method
        simply wraps the function, and so the docstring for ivy.device.dev_clone_array
        also applies to this method with minimal changes.

        Examples
        --------

        """
        return self.static_dev_clone_array(self, devices)
