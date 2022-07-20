# global
import abc
from typing import Union, Optional, Any

import ivy


# ToDo: implement all methods here as public instance methods


class ArrayWithDevice(abc.ABC):
    def dev(self: ivy.Array) -> Union[ivy.Device, ivy.NativeDevice]:
        """
        ivy.Array instance method variant of ivy.dev. This method simply wraps
        the function, and so the docstring for ivy.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return ivy.dev(self)

    def to_device(
        self: ivy.Array,
        device: Union[ivy.Device, ivy.NativeDevice],
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.to_device. This method simply
        wraps the function, and so the docstring for ivy.to_device also applies
        to this method with minimal changes.

        Examples
        --------

        """
        return ivy.to_device(self, device, stream=stream, out=out)
