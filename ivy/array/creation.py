# global
import abc
from typing import Optional, Union, List

# local
import ivy

# Array API Standard #
# -------------------#


class ArrayWithCreation(abc.ABC):
    def asarray(
        self: ivy.Array,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Array:
        return ivy.asarray(self._data, copy=copy, dtype=dtype, device=device)

    def full_like(
        self: ivy.Array,
        fill_value: Union[int, float],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.full_like(
            self._data, fill_value, dtype=dtype, device=device, out=out
        )

    def ones_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.ones_like(self._data, dtype=dtype, device=device, out=out)

    def zeros_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.zeros_like(self._data, dtype=dtype, device=device, out=out)

    def tril(self: ivy.Array, k: int = 0, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.tril(self._data, k, out=out)

    def triu(self: ivy.Array, k: int = 0, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.triu(self._data, k, out=out)

    def empty_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.empty_like(self._data, dtype=dtype, device=device, out=out)

    def meshgrid(
        *arrays: Union[ivy.Array, ivy.NativeArray], indexing: Optional[str] = "xy"
    ) -> List[ivy.Array]:
        return ivy.meshgrid(*arrays, indexing=indexing)

    def from_dlpack(
        self: ivy.Array,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.from_dlpack(self._data, out=out)

    # Extra #
    # ------#

    def native_array(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.NativeArray:
        return ivy.to_native(ivy.asarray(self._data, dtype=dtype, device=device))
