"""
Function stubs for set functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/set_functions.md
"""

from __future__ import annotations

from ._types import Tuple, array

def unique_all(x: array, /) -> Tuple[array, array, array, array]:
    pass

def unique_counts(x: array, /) -> Tuple[array, array]:
    pass

def unique_inverse(x: array, /) -> Tuple[array, array]:
    pass

def unique_values(x: array, /) -> array:
    pass

__all__ = ['unique_all', 'unique_counts', 'unique_inverse', 'unique_values']
