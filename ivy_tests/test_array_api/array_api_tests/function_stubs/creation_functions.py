"""
Function stubs for creation functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/creation_functions.md
"""

from __future__ import annotations

from ._types import (List, NestedSequence, Optional, SupportsBufferProtocol, Tuple, Union, array,
                     device, dtype)

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def asarray(obj: Union[array, bool, int, float, NestedSequence[bool|int|float], SupportsBufferProtocol], /, *, dtype: Optional[dtype] = None, device: Optional[device] = None, copy: Optional[bool] = None) -> array:
    pass

def empty(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def empty_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def eye(n_rows: int, n_cols: Optional[int] = None, /, *, k: int = 0, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def from_dlpack(x: object, /) -> array:
    pass

def full(shape: Union[int, Tuple[int, ...]], fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def full_like(x: array, /, fill_value: Union[int, float], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def linspace(start: Union[int, float], stop: Union[int, float], /, num: int, *, dtype: Optional[dtype] = None, device: Optional[device] = None, endpoint: bool = True) -> array:
    pass

def meshgrid(*arrays: array, indexing: str = 'xy') -> List[array, ...]:
    pass

def ones(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def ones_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def tril(x: array, /, *, k: int = 0) -> array:
    pass

def triu(x: array, /, *, k: int = 0) -> array:
    pass

def zeros(shape: Union[int, Tuple[int, ...]], *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

def zeros_like(x: array, /, *, dtype: Optional[dtype] = None, device: Optional[device] = None) -> array:
    pass

__all__ = ['arange', 'asarray', 'empty', 'empty_like', 'eye', 'from_dlpack', 'full', 'full_like', 'linspace', 'meshgrid', 'ones', 'ones_like', 'tril', 'triu', 'zeros', 'zeros_like']
