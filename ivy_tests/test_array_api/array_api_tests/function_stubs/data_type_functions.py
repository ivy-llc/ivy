"""
Function stubs for data type functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/data_type_functions.md
"""

from __future__ import annotations

from ._types import List, Tuple, Union, array, dtype, finfo_object, iinfo_object

def astype(x: array, dtype: dtype, /, *, copy: bool = True) -> array:
    pass

def broadcast_arrays(*arrays: array) -> List[array]:
    pass

def broadcast_to(x: array, /, shape: Tuple[int, ...]) -> array:
    pass

def can_cast(from_: Union[dtype, array], to: dtype, /) -> bool:
    pass

def finfo(type: Union[dtype, array], /) -> finfo_object:
    pass

def iinfo(type: Union[dtype, array], /) -> iinfo_object:
    pass

def result_type(*arrays_and_dtypes: Union[array, dtype]) -> dtype:
    pass

__all__ = ['astype', 'broadcast_arrays', 'broadcast_to', 'can_cast', 'finfo', 'iinfo', 'result_type']
