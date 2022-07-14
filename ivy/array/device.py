# global
import abc
from typing import Union

import ivy


# ToDo: implement all methods here as public instance methods

class ArrayWithDevice(abc.ABC):
    def dev(self: ivy.Array) -> Union[ivy.Device, ivy.NativeDevice]:
        """
        ivy.Array instance method variant of ivy.device.dev. This method simply wraps
        the function, and so the docstring for ivy.device.dev also applies to this
        method with minimal changes.

        Examples
        --------

        """
        return ivy.device.dev(self)
