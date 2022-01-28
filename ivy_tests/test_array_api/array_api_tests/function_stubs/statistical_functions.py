"""
Function stubs for statistical functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/statistical_functions.md
"""

from __future__ import annotations

from ._types import Optional, Tuple, Union, array, dtype

def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    pass

def mean(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    pass

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    pass

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[dtype] = None, keepdims: bool = False) -> array:
    pass

def std(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    pass

def sum(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[dtype] = None, keepdims: bool = False) -> array:
    pass

def var(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, correction: Union[int, float] = 0.0, keepdims: bool = False) -> array:
    pass

__all__ = ['max', 'mean', 'min', 'prod', 'std', 'sum', 'var']
