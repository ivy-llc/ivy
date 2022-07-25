# global
import abc
from typing import Optional, Union

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithRandom(abc.ABC):
    def random_uniform(
        self: ivy.Array,
        high: Union[ivy.Array, ivy.NativeArray] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.random_uniform(
            self._data,
            high,
            shape,
            device=device,
            dtype=dtype,
            out=out,
        )
