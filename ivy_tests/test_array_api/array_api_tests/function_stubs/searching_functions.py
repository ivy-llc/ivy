"""
Function stubs for searching functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/searching_functions.md
"""

from __future__ import annotations

from ._types import Optional, Tuple, array

def argmax(x: array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> array:
    pass

def argmin(x: array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> array:
    pass

def nonzero(x: array, /) -> Tuple[array, ...]:
    pass

def where(condition: array, x1: array, x2: array, /) -> array:
    pass

__all__ = ['argmax', 'argmin', 'nonzero', 'where']
