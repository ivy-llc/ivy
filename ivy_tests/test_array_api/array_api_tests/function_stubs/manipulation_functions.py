"""
Function stubs for manipulation functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/manipulation_functions.md
"""

from __future__ import annotations

from ._types import List, Optional, Tuple, Union, array

def concat(arrays: Union[Tuple[array, ...], List[array]], /, *, axis: Optional[int] = 0) -> array:
    pass

def expand_dims(x: array, /, *, axis: int) -> array:
    pass

def flip(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    pass

def permute_dims(x: array, /, axes: Tuple[int, ...]) -> array:
    pass

def reshape(x: array, /, shape: Tuple[int, ...]) -> array:
    pass

def roll(x: array, /, shift: Union[int, Tuple[int, ...]], *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> array:
    pass

def squeeze(x: array, /, axis: Union[int, Tuple[int, ...]]) -> array:
    pass

def stack(arrays: Union[Tuple[array, ...], List[array]], /, *, axis: int = 0) -> array:
    pass

__all__ = ['concat', 'expand_dims', 'flip', 'permute_dims', 'reshape', 'roll', 'squeeze', 'stack']
