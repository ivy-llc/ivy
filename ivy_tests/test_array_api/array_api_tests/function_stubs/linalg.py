"""
Function stubs for linear algebra functions (Extension).

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/linear_algebra_functions.md
"""

from __future__ import annotations

from ._types import Literal, Optional, Tuple, Union, array
from .constants import inf
from collections.abc import Sequence

def cholesky(x: array, /, *, upper: bool = False) -> array:
    pass

def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    pass

def det(x: array, /) -> array:
    pass

def diagonal(x: array, /, *, offset: int = 0) -> array:
    pass

def eigh(x: array, /) -> Tuple[array]:
    pass

def eigvalsh(x: array, /) -> array:
    pass

def inv(x: array, /) -> array:
    pass

def matmul(x1: array, x2: array, /) -> array:
    pass

def matrix_norm(x: array, /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal[inf, -inf, 'fro', 'nuc']]] = 'fro') -> array:
    pass

def matrix_power(x: array, n: int, /) -> array:
    pass

def matrix_rank(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    pass

def matrix_transpose(x: array, /) -> array:
    pass

def outer(x1: array, x2: array, /) -> array:
    pass

def pinv(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    pass

def qr(x: array, /, *, mode: Literal['reduced', 'complete'] = 'reduced') -> Tuple[array, array]:
    pass

def slogdet(x: array, /) -> Tuple[array, array]:
    pass

def solve(x1: array, x2: array, /) -> array:
    pass

def svd(x: array, /, *, full_matrices: bool = True) -> Union[array, Tuple[array, ...]]:
    pass

def svdvals(x: array, /) -> array:
    pass

def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> array:
    pass

def trace(x: array, /, *, offset: int = 0) -> array:
    pass

def vecdot(x1: array, x2: array, /, *, axis: int = None) -> array:
    pass

def vector_norm(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Union[int, float, Literal[inf, -inf]] = 2) -> array:
    pass

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']
