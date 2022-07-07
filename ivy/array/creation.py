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
        """
        ivy.Array instance method variant of ivy.asarray. This method simply wraps the
        function, and so the docstring for ivy.asarray also applies to this method
        with minimal changes.
        """
        return ivy.asarray(self._data, copy=copy, dtype=dtype, device=device)

    def full_like(
        self: ivy.Array,
        fill_value: float,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.full_like. This method simply wraps the
        function, and so the docstring for ivy.full_like also applies to this method
        with minimal changes.
        """
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
        """
        ivy.Array instance method variant of ivy.ones_like. This method simply wraps the
        function, and so the docstring for ivy.ones_like also applies to this method
        with minimal changes.
        """
        return ivy.ones_like(self._data, dtype=dtype, device=device, out=out)

    def zeros_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.zeros_like. This method simply wraps
        the function, and so the docstring for ivy.zeros_like also applies to this
        method with minimal changes.
        """
        return ivy.zeros_like(self._data, dtype=dtype, device=device, out=out)

    def tril(self: ivy.Array, k: int = 0, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.tril. This method simply wraps the
        function, and so the docstring for ivy.tril also applies to this method
        with minimal changes.
        """
        return ivy.tril(self._data, k, out=out)

    def triu(self: ivy.Array, k: int = 0, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.triu. This method simply wraps the
        function, and so the docstring for ivy.triu also applies to this method
        with minimal changes.
        """
        return ivy.triu(self._data, k, out=out)

    def empty_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.empty_like. This method simply wraps
        the function, and so the docstring for ivy.empty_like also applies to this
        method with minimal changes.
        """
        return ivy.empty_like(self._data, dtype=dtype, device=device, out=out)

    def meshgrid(
        self: ivy.Array,
        *arrays: Union[ivy.Array, ivy.NativeArray],
        indexing: Optional[str] = "xy",
    ) -> List[ivy.Array]:
        list_arrays = [self._data] + list(arrays)
        """
        ivy.Array instance method variant of ivy.meshgrid. This method simply wraps the
        function, and so the docstring for ivy.meshgrid also applies to this method
        with minimal changes.
        """
        return ivy.meshgrid(*list_arrays, indexing=indexing)

    def from_dlpack(
        self: ivy.Array,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.from_dlpack. This method simply wraps
        the function, and so the docstring for ivy.from_dlpack also applies to this
        method with minimal changes.
        """
        return ivy.from_dlpack(self._data, out=out)

    # Extra #
    # ------#

    def native_array(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.NativeArray:
        """
        ivy.Array instance method variant of ivy.native_array. This method simply wraps
        the function, and so the docstring for ivy.native_array also applies to this
        method with minimal changes.
        """
        return ivy.native_array(self._data, dtype=dtype, device=device)
