# global
import abc
from typing import Optional, Union, Tuple, List, Iterable
from numbers import Number
import numpy as np

# local
import ivy

# Array API Standard #
# -------------------#

class ArrayWithCreation(abc.ABC):
    def arange(
        self: ivy.Array,
        start: Number,
        stop: Optional[Number] = None,
        step: Number = 1,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        ) -> ivy.array:
        return ivy.arange(
            start, stop, step, dtype=dtype, device=device, out=out
            )
    
    def asarray(
        self: ivy.Array,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        ) -> ivy.Array:
        return ivy.asarray([self._data], copy=copy, dtype=dtype, device=device)
    
    def zeros(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.zeros(shape, dtype=dtype, device=device, out=out)

    def ones(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.ones(shape, dtype=dtype, device=device, out=out)

    def full_like(
        self: ivy.Array,
        fill_value: Union[int, float],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.full_like(
            [self._data], fill_value, dtype=dtype, device=device, out=out
            )
    
    def ones_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.ones_like([self._data], dtype=dtype, device=device, out=out)

    def zeros_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.zeros_like([self._data], dtype=dtype, device=device, out=out)

    def tril(
        self: ivy.Array,
        k: int = 0,
        out: Optional[ivy.Array] = None
        ) -> ivy.Array:
        return ivy.tril(self, k, out=out)
    
    def triu(
        self: ivy.Array,
        k: int = 0,
        out: Optional[ivy.Array] = None
        ) -> ivy.Array:
        return ivy.triu(self, k, out=out)

    def empty(
        shape: Union[int, Tuple[int], List[int]],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.empty(shape, dtype=dtype, device=device, out=out)
    
    def empty_like(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.empty_like(self, dtype=dtype, device=device, out=out)

    def eye(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.eye(n_rows, n_cols, k, dtype=dtype, device=device, out=out)

    def linspace(
        start: Union[ivy.Array, ivy.NativeArray, int, float],
        stop: Union[ivy.Array, ivy.NativeArray, int, float],
        num: int,
        axis: int = None,
        endpoint: bool = True,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> Union[ivy.Array, ivy.NativeArray]:
        return ivy.linspace(
            start, 
            stop, 
            num, 
            axis, 
            endpoint=endpoint, 
            dtype=dtype, 
            device=device, 
            out=out
            )

    def meshgrid(
        *arrays: Union[ivy.Array, ivy.NativeArray],
        indexing: Optional[str] = "xy"
        ) -> List[ivy.Array]:
        return ivy.meshgrid(*arrays, indexing=indexing)

    def full(
        shape: Union[int, Tuple[int, ...]],
        fill_value: Union[int, float],
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.full(
            shape, fill_value, dtype=dtype, device=device, out=out
            )
    
    def from_dlpack(
        self: ivy.Array,
        out: Optional[ivy.Array] = None,
        ) -> ivy.Array:
        return ivy.from_dlpack(self, out=out)


# Extra #
# ------#

    def native_array(
        self: ivy.Array,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        ) -> ivy.NativeArray:
        return ivy.to_native(ivy.asarray(self, dtype=dtype, device=device))

    def logspace(
        start: Union[ivy.Array, ivy.NativeArray, int],
        stop: Union[ivy.Array, ivy.NativeArray, int],
        num: int,
        base: float = 10.0,
        axis: int = None,
        *,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        out: Optional[ivy.Array] = None,
        ) -> Union[ivy.Array, ivy.NativeArray]:
        return ivy.logspace(start, stop, num, base, axis, device=device, out=out)









