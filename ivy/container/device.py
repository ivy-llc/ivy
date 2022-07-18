# local
from typing import Union, Optional, Any, Iterable

import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods

# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def dev_static(container: ivy.Container) -> ivy.Container:
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
        return self.dev_static(self)

    @staticmethod
    def to_device_static(
        container: ivy.Container,
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
            "to_device", container, device, stream=stream, out=out
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
        return self.to_device_static(self, device, stream=stream, out=out)

    @staticmethod
    def dev_clone_array_static(
        container: ivy.Container, devices: Iterable[ivy.Device]
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.device.dev_clone_array. This method
        simply wraps the function, and so the docstring for ivy.device.dev_clone_array
        also applies to this method with minimal changes.

        Examples
        --------

        """
        return ContainerBase.multi_map_in_static_method(
            "dev_clone_array", container, devices
        )

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
        return self.dev_clone_array_static(self, devices)
